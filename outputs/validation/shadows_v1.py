"""Ground-plane occlusion from tire supports and the lower chassis boundary."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def _polygon(shape, points):
    # Project a rounded chassis footprint, with a smooth density taper. Sharp
    # quadrilateral corners produce an implausible rectangular black carpet.
    n=256
    yy,xx=np.mgrid[:n,:n].astype(np.float32)
    u=(xx-(n-1)/2)/((n-1)/2)
    v=(yy-(n-1)/2)/((n-1)/2)
    radius=(np.abs(u)**3.2+np.abs(v)**3.2)**(1/3.2)
    t=np.clip((radius-.64)/.36,0,1)
    density=1-t*t*(3-2*t)
    corners=np.array([[0,n-1],[n-1,n-1],[n-1,0],[0,0]],np.float32)
    transform=cv2.getPerspectiveTransform(corners,np.asarray(points,np.float32))
    return cv2.warpPerspective(density,transform,(shape[1],shape[0]),flags=cv2.INTER_LINEAR)


def _blur(mask, sx, sy):
    return cv2.GaussianBlur(mask, (0,0), sigmaX=max(.5,float(sx)),
                            sigmaY=max(.5,float(sy)), borderType=cv2.BORDER_CONSTANT)


def _ellipse(shape, x, y, sx, sy, strength):
    yy,xx=np.ogrid[:shape[0],:shape[1]]
    return (strength*np.exp(-.5*(((xx-x)/max(sx,.5))**2 + ((yy-y)/max(sy,.5))**2))).astype(np.float32)


def render_shadows(shape: tuple[int,int], geometry: dict) -> dict[str,np.ndarray]:
    """Return independent opacity layers (float32 0..1), composited as light transmission.

    Tire shadows have the highest spatial frequency. The chassis occludes a
    projected footprint with a short penumbra; a weaker broad component models
    diffuse environment occlusion. Hidden supports receive no artificial sharp
    tire shadow. Parameters scale with estimated tire size.
    """
    contacts=geometry['contacts']
    radius=float(np.median([c['radius'] for c in contacts]))
    x0,y0,x1,y1=geometry['bbox']
    bw,bh=x1-x0,y1-y0
    footprint=np.asarray(geometry.get('chassis_footprint',geometry['footprint']),np.float32)
    if 'chassis_footprint' not in geometry:
        # Extend the axle support polygon along its longitudinal axis to cover
        # bumper overhang; this still remains a four-sided ground footprint.
        left,right=footprint[0].copy(),footprint[1].copy()
        dx=max(1.,right[0]-left[0]); slope=(right[1]-left[1])/dx
        extension_l=max(0,left[0]-(x0+.03*bw))
        extension_r=max(0,(x1-.03*bw)-right[0])
        footprint[0] -= np.array([extension_l,extension_l*slope])
        footprint[3] -= np.array([extension_l,extension_l*slope])
        footprint[1] += np.array([extension_r,extension_r*slope])
        footprint[2] += np.array([extension_r,extension_r*slope])
    # Bring the near penumbra just beyond the support line. Side views need a
    # much thinner visible ground projection than an end-on view.
    end_on=geometry['view_cues']['support_mode']=='occluded_axle_pair'
    near_spill=radius*(.52 if end_on else .08)
    footprint[:2,1] += near_spill
    core=_polygon(shape,footprint)
    # The lower chassis supplies an occlusion boundary, while the tire line
    # supplies the receiving plane. This fills clearance under bumpers/sills
    # that a rounded axle rectangle alone misses at strong three-quarter views.
    # Only this lower boundary is used, never a shifted full vehicle silhouette.
    apron=np.zeros(shape,np.float32)
    profile=np.asarray(geometry.get('lower_body_profile',[]),np.float32)
    if len(profile):
        xs=np.arange(max(0,int(np.ceil(profile[0,0]))),min(shape[1],int(profile[-1,0])+1))
        lower=np.interp(xs,profile[:,0],profile[:,1])
        left,right=contacts[0],contacts[-1]
        slope=(right['y']-left['y'])/max(1,right['x']-left['x'])
        plane=left['y']+slope*(xs-left['x'])
        outside=(xs<left['x'])|(xs>right['x'])
        plane[outside]=np.minimum(plane[outside],lower[outside]+radius*.32)
        if end_on:
            u=(xs-(left['x']+right['x'])/2)/max(1,(right['x']-left['x'])*.55)
            plane += radius*.63*np.sqrt(np.maximum(0,1-u*u))
        else:
            plane += radius*.035
        plane=gaussian_filter1d(plane,max(1,radius*.10))
        yy=np.arange(shape[0],dtype=np.float32)[:,None]
        top=lower-radius*.18
        density=((yy>=top[None,:])&(yy<=plane[None,:])).astype(np.float32)
        # Suppress high isolated features (for example a mirror near a bounding
        # box extreme), which are not evidence of chassis-ground clearance.
        density[:,lower<max(y0+.63*bh,min(c['y'] for c in contacts)-radius*.45)]=0
        if end_on:
            density[:,np.abs(u)>1]=0
        else:
            edge=np.minimum(xs-x0,x1-xs)
            lateral=np.clip(edge/max(1,radius*.70),0,1)
            density *= lateral[None,:]
        apron[:,xs]=density
    support=1-(1-core*.68)*(1-apron)
    underbody=.86*_blur(support, radius*.13, radius*(.18 if end_on else .09))
    broad=.32*_blur(support, radius*.52, radius*(.48 if end_on else .24))
    tire=np.zeros(shape,np.float32)
    for c in contacts:
        r=c['radius']
        local=_ellipse(shape,c['x'],c['y']+.5,max(c['half_width']*1.2,r*.27),
                       max(1.,r*.065),.86)
        tire=1-(1-tire)*(1-local)
    combined=1-(1-underbody)*(1-broad)*(1-tire)
    return {'tire_contact':tire, 'underbody':underbody, 'ground':broad,
            'combined':np.clip(combined,0,.97).astype(np.float32)}


def composite_white(rgb:np.ndarray, alpha:np.ndarray, shadow:np.ndarray|None=None)->np.ndarray:
    a=alpha.astype(np.float32)[...,None]/255
    ground=np.full(alpha.shape,255,np.float32) if shadow is None else 255*(1-shadow)
    return np.uint8(np.clip(np.rint(rgb*a+ground[...,None]*(1-a)),0,255))
