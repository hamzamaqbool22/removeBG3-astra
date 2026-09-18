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


OBLIQUE_PARAMETERS = {
    'side_plane_lift': .217,
    'end_center_fraction': .361,
    'end_width_fraction': .488,
    'end_center_drop': .122,
    'end_ground_spill': .055,
    'end_tilt_fraction': .035,
    'core_opacity': .897,
    'core_blur_x': .209,
    'core_blur_y': .076,
    'ambient_opacity': .045,
}


def _oblique_support(shape, geometry, params):
    """Separate the longitudinal chassis and the visible end's ground lobe.

    Wheel contacts lie along the *near* side of the car. The sill's shadow
    recedes inside that line, while the bumper occludes a broader end footprint.
    A smooth lobe avoids reproducing exhaust/hidden-wheel bumps in the shadow.
    The same rule handles either end and either direction without view labels.
    """
    contacts = geometry['contacts']
    left, right = contacts[0], contacts[-1]
    x0, y0, x1, y1 = geometry['bbox']
    left_gap, right_gap = left['x'] - x0, x1 - right['x']
    end_left = left_gap > right_gap
    near = left if end_left else right
    overhang = left_gap if end_left else right_gap
    direction = -1 if end_left else 1
    radius = near['radius']
    profile = np.asarray(geometry['lower_body_profile'], np.float32)
    xs = np.arange(shape[1], dtype=np.float32)
    lower = np.interp(xs, profile[:, 0], profile[:, 1])
    yy = np.arange(shape[0], dtype=np.float32)[:, None]

    slope = (right['y'] - left['y']) / max(1, right['x'] - left['x'])
    plane = left['y'] + slope * (xs - left['x'])
    shift_x, shift_y = geometry['view_cues']['opposite_track_shift']
    plane += shift_y * params['side_plane_lift']
    top = lower - .18 * radius
    band = ((yy >= top[None, :]) & (yy <= plane[None, :])).astype(np.float32)
    extension = .18 * radius
    band[:, (xs < left['x'] - extension) | (xs > right['x'] + extension)] = 0
    # Taper rather than abruptly ending the occlusion at the axles.
    taper = np.clip(np.minimum(xs-left['x']+extension,
                               right['x']+extension-xs) / max(1, .4*radius), 0, 1)
    band *= taper[None, :]

    # Estimate bumper clearance from a robust lower-body statistic, excluding
    # the near tire. Individual exhaust tips/far tires cannot create lobes.
    distance_into_end = direction * (profile[:, 0] - near['x'])
    valid = (distance_into_end > .30*overhang) & (distance_into_end < .84*overhang)
    bumper_y = float(np.median(profile[valid, 1])) if np.any(valid) else near['y']-.6*radius
    center_x = near['x'] + direction * overhang * params['end_center_fraction']
    center_y = bumper_y + radius * params['end_center_drop']
    rx = max(radius, overhang * params['end_width_fraction'])
    # Anchor the lobe's ground extent to the actual tire contact. An ellipse
    # sized only from a detected wheel radius floats upward when the far-side
    # wheel is underestimated or the bumper has unusually high clearance.
    ry = float(np.clip(near['y'] + radius*params['end_ground_spill'] - center_y,
                       radius*.30, radius*.95))
    tilt = params['end_tilt_fraction'] * shift_y / (shift_x if abs(shift_x)>1 else direction)
    u = (xs-center_x) / rx
    v = (yy-center_y-tilt*(xs-center_x)[None, :]) / ry
    distance = np.sqrt(u[None, :]**2 + v**2)
    edge = np.clip((distance-.93)/.12, 0, 1)
    end = (1-edge*edge*(3-2*edge)).astype(np.float32)
    support = 1-(1-band)*(1-end)
    return support, radius, end


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
    # Preserve the end-on rendering. Only a clear
    # projected track separation activates the three-quarter correction.
    oblique_weight = 0. if end_on else float(np.clip(
        (geometry['view_cues']['perspective_score']-.15)/.55, 0, 1))
    oblique_weight *= float(np.clip((abs(geometry['view_cues']['contact_slope'])-.035)/.06,0,1))
    end_layer = np.zeros(shape, np.float32)
    if oblique_weight > 0:
        p = OBLIQUE_PARAMETERS
        support_q, near_radius, end = _oblique_support(shape, geometry, p)
        core_q = p['core_opacity'] * _blur(support_q, near_radius*p['core_blur_x'], near_radius*p['core_blur_y'])
        broad_q = p['ambient_opacity'] * _blur(support_q, near_radius*.52, near_radius*.24)
        underbody = underbody*(1-oblique_weight)+core_q*oblique_weight
        broad = broad*(1-oblique_weight)+broad_q*oblique_weight
        end_layer = p['core_opacity'] * _blur(end, near_radius*p['core_blur_x'], near_radius*p['core_blur_y']) * oblique_weight
    # Profile views need a continuous chassis occlusion band. The generic
    # apron rejects the sill as "too high" when clearance exceeds .45 radii,
    # leaving only a weak rounded footprint between the wheels.
    side_weight = 0. if end_on else float(np.clip(
        (geometry['view_cues']['side_on_score'] - .55) / .20, 0, 1))
    side_weight *= float(np.clip((.06 - abs(geometry['view_cues']['contact_slope'])) / .025, 0, 1))
    if side_weight > 0 and len(profile):
        xs = np.arange(shape[1], dtype=np.float32)
        yy = np.arange(shape[0], dtype=np.float32)[:, None]
        lower = np.interp(xs, profile[:, 0], profile[:, 1])
        left, right = contacts[0], contacts[-1]
        plane = left['y'] + (right['y']-left['y']) * (xs-left['x']) / max(1, right['x']-left['x'])
        # Recede slightly behind the tire contact line; fill bumper overhangs
        # as well as the space between axles, with a rounded lateral taper.
        end_rounding = 1 - np.clip(np.minimum(xs-x0, x1-xs) / max(1, 1.1*radius), 0, 1)
        bottom = plane - .08 * radius - .38*radius*end_rounding**2
        outside = (xs < left['x']) | (xs > right['x'])
        bottom[outside] = np.minimum(bottom[outside], lower[outside] + .24*radius)
        bottom = gaussian_filter1d(bottom, max(1, radius*.16))
        top = np.minimum(lower - .16 * radius, bottom - .25 * radius)
        band = ((yy >= top[None, :]) & (yy <= bottom[None, :])).astype(np.float32)
        edge = np.clip(np.minimum(xs-x0, x1-xs) / max(1, .95*radius), 0, 1)
        band *= (edge*edge*(3-2*edge))[None, :]
        side_core = .92 * _blur(band, radius*.19, radius*.10)
        side_ground = .18 * _blur(band, radius*.48, radius*.22)
        underbody = underbody*(1-side_weight) + side_core*side_weight
        broad = broad*(1-side_weight) + side_ground*side_weight
    tire=np.zeros(shape,np.float32)
    for c in contacts:
        r=c['radius']
        local=_ellipse(shape,c['x'],c['y']+.5,max(c['half_width']*1.2,r*.27),
                       max(1.,r*.065),.86)
        tire=1-(1-tire)*(1-local)
    combined=1-(1-underbody)*(1-broad)*(1-tire)
    return {'tire_contact':tire, 'underbody':underbody, 'ground':broad, 'end_footprint':end_layer,
            'combined':np.clip(combined,0,.97).astype(np.float32)}


def composite_white(rgb:np.ndarray, alpha:np.ndarray, shadow:np.ndarray|None=None)->np.ndarray:
    a=alpha.astype(np.float32)[...,None]/255
    ground=np.full(alpha.shape,255,np.float32) if shadow is None else 255*(1-shadow)
    return np.uint8(np.clip(np.rint(rgb*a+ground[...,None]*(1-a)),0,255))
