"""Recover visible cast-shadow opacity from source pavement using CPU image processing.

No model or angle labels. A robust smooth ground estimate separates broad lighting
attenuation from surface texture. Unreliable scenes return None to use the renderer.
"""
import cv2
import numpy as np
from scipy.ndimage import median_filter


def recover_shadow(rgb, alpha):
    height, width = alpha.shape
    scale = min(1., 1000 / max(height, width))
    size = (round(width*scale), round(height*scale))
    color = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA).astype(np.float32)/255
    mask = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA) > 127
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    x0,x1,y0,y1 = xs.min(),xs.max(),ys.min(),ys.max()
    bw,bh = x1-x0+1,y1-y0+1
    h,w = mask.shape
    yy,xx = np.mgrid[:h,:w]
    bottom = np.max(np.where(mask, yy, -1), axis=0)
    floor = np.interp(np.arange(w), np.arange(x0,x1+1), bottom[x0:x1+1])
    floor = np.maximum(floor, y0+.72*bh)
    # Only ground below the lower vehicle edge; never roof/sky, wall or windows.
    roi = ((xx > x0-.22*bw) & (xx < x1+.22*bw) &
           (yy >= floor[None,:]-.18*bh) & (yy < floor[None,:]+.48*bh) & ~mask)
    distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    lum = color @ np.array([.2126,.7152,.0722],np.float32)
    smooth = cv2.GaussianBlur(lum,(0,0),max(.8,bw*.0015))
    # Repeated floor tiles are reflectance, not holes in the shadow. Detect
    # distributed contrast away from the car, then estimate illumination at a
    # coarser scale only on those floors. Normalized convolution excludes car
    # pixels: a dark body must never bleed into a manufactured shadow.
    ground = (~mask).astype(np.float32)
    sigma = max(1., bw*.009)
    coarse = cv2.GaussianBlur(lum*ground,(0,0),sigma) / np.maximum(
        cv2.GaussianBlur(ground,(0,0),sigma), 1e-6)
    texture_region = roi & (distance > .10*bh) & (yy > y0+.80*bh)
    texture = np.abs(smooth-coarse)[texture_region]
    patterned = len(texture) > 300 and np.mean(texture > .035) > .30
    if patterned:
        smooth = coarse
    # Fit low-frequency pavement illumination; trim dark outliers (shadow,
    # cracks) and bright outliers (painted stripes) rather than transferring them.
    samples = roi & (distance > .07*bh) & (np.ptp(color,axis=2)<.24) & (lum>.12) & (lum<.94)
    sy,sx = np.where(samples)
    if len(sx)<300:
        return None
    step=max(1,len(sx)//16000);sy,sx=sy[::step],sx[::step]
    xn=(sx-x0)/bw;yn=(sy-y1)/bh
    design=np.stack([np.ones_like(xn),xn,yn,xn*yn,xn*xn,yn*yn],axis=1)
    values=smooth[sy,sx]
    valid=values>=np.quantile(values,.45)
    for _ in range(5):
        coeff=np.linalg.lstsq(design[valid],values[valid],rcond=None)[0]
        residual=values-design@coeff
        spread=max(.018,float(np.median(np.abs(residual[valid])))*2.5)
        valid=(residual > -spread)&(residual < spread)
        if valid.sum()<150:
            return None
    xn=(xx-x0)/bw;yn=(yy-y1)/bh
    base=(coeff[0]+coeff[1]*xn+coeff[2]*yn+coeff[3]*xn*yn+coeff[4]*xn*xn+coeff[5]*yn*yn)
    if np.median(np.abs(values[valid]-design[valid]@coeff))>.055:
        return None
    base=np.clip(base,.12,.95)
    opacity=np.clip(1-smooth/base,0,.94).astype(np.float32)
    # OpenCV requires uint8 for a 7x7 median; preserve float opacity and
    # its precision with SciPy. nearest matches OpenCV's replicated borders.
    # Dark floor seams can cross an otherwise valid shadow and become connected
    # to it. Remove narrow opacity ridges, not just disconnected components.
    # Grayscale opening preserves the broad shadow underneath those ridges.
    seam_size=max(11,int(round(.025*min(bw,bh))) | 1)
    seam_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(seam_size,seam_size))
    opacity=cv2.morphologyEx(opacity,cv2.MORPH_OPEN,seam_kernel)
    opacity=median_filter(opacity,size=7,mode="nearest")
    opacity=cv2.bilateralFilter(opacity,9,.12,4)
    opacity=cv2.GaussianBlur(opacity,(0,0),max(.7,bw*.0012))
    candidate=((opacity>.16)&roi).astype(np.uint8)
    # Frame borders are often black screenshot/crop lines, not cast shadows.
    candidate[:3]=0; candidate[-3:]=0; candidate[:,:3]=0; candidate[:,-3:]=0
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    candidate=cv2.morphologyEx(candidate,cv2.MORPH_CLOSE,kernel)
    candidate=cv2.morphologyEx(candidate,cv2.MORPH_OPEN,kernel)
    count,labels,stats,_=cv2.connectedComponentsWithStats(candidate,8)
    accepted=np.zeros_like(candidate)
    for i in range(1,count):
        component=labels==i
        area=stats[i,cv2.CC_STAT_AREA]
        if area<max(30,bw*bh*.001) or area>bw*bh*.5:
            continue
        attached=component & (distance<max(5,.04*bh))
        if attached.sum()<max(6,.007*bw):
            continue
        accepted[component]=1
    if accepted.sum()<bw*bh*.008:
        return None
    # Close small pavement markings inside the shadow, preserving its contour.
    opacity=cv2.morphologyEx(opacity,cv2.MORPH_CLOSE,kernel)
    support=cv2.GaussianBlur(accepted.astype(np.float32),(0,0),max(.65,bw*.001))
    if patterned:
        # The .16 candidate threshold locates connected shadow cores, not the
        # penumbra boundary. Recover the low-opacity tails around those cores
        # instead of cutting them off at every bright tile. This support is
        # still gated by measured attenuation and the plausible ground ROI.
        radius = max(2, int(round(.022*bh)))
        expanded = cv2.dilate(accepted, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))).astype(np.float32)
        support = cv2.GaussianBlur(expanded,(0,0),max(1.,bw*.006))
        support *= (roi | mask)
    recovered=opacity*support
    # Missing pixels outside a cropped source cannot be reconstructed. Taper
    # that boundary instead of exporting an artificial straight shadow edge.
    edge_distance=np.minimum.reduce([xx,yy,w-1-xx,h-1-yy])
    fade=np.clip(edge_distance/max(3,.012*bw),0,1)
    recovered*=fade*fade*(3-2*fade)
    # Carry opacity a few pixels underneath the cutout to avoid resampling seams.
    inside=cv2.dilate(recovered,np.ones((5,5),np.uint8))
    recovered[mask]=inside[mask]
    return cv2.resize(recovered,(width,height),interpolation=cv2.INTER_LINEAR)
