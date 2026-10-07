"""Source-preserving CPU shadows with selective weak-ground completion."""
import cv2
import numpy as np
from vehicle_pipeline.contact_shadow import complete_contact_shadow
from vehicle_pipeline.source_shadow import recover_shadow


def expand_ground_shadow(alpha, shadow, amount=.012):
    """Add a small soft outer margin without darkening the source core.

    Expand left/right and along the pavement, with less vertical spread. The
    lower silhouette gates the added layer so it cannot climb the body sides.
    Amount is relative to vehicle dimensions, not image resolution.
    """
    if not np.isfinite(amount) or not 0 <= amount <= .03:
        raise ValueError('Expansion must be between 0 and .03')
    mask = alpha > 127
    ys, xs = np.where(mask)
    if not len(xs) or amount == 0:
        return shadow.copy()
    bw, bh = np.ptp(xs)+1, np.ptp(ys)+1
    rx, ry = max(1,round(amount*bw)), max(1,round(amount*bh))
    # Opaque car pixels carry no visible source evidence and must not seed an
    # expansion. Source recovery may store values there for resampling seams.
    visible = np.where(mask,0,shadow).astype(np.float32)
    if not np.any(visible):
        return shadow.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rx+1,2*ry+1))
    margin = cv2.dilate(visible,kernel)
    margin = cv2.GaussianBlur(margin,(0,0),sigmaX=max(.6,rx*.45),sigmaY=max(.6,ry*.45))
    yy = np.arange(mask.shape[0])[:,None]
    bottom = np.max(np.where(mask,yy,-1),axis=0)
    valid = np.flatnonzero(mask.any(axis=0))
    floor = np.interp(np.arange(mask.shape[1]),valid,bottom[valid])
    gate = np.clip((yy-floor[None,:])/max(1.,ry),0,1)
    margin *= gate*gate*(3-2*gate)
    margin[mask] = 0
    return np.maximum(shadow,.72*margin).astype(np.float32)


def deepen_weak_footprint(alpha, footprint):
    """Extend a weak ground footprint outward from its contact contour.

    Stretch only ground depth, preserving tire attachment and lateral extent.
    This is neutral grounding, not an inferred directional sunlight shadow.
    """
    mask = alpha > 127
    if not mask.any():
        return footprint.copy()
    yy, xx = np.indices(mask.shape, dtype=np.float32)
    bottom = np.max(np.where(mask, yy, -1), axis=0)
    valid = np.flatnonzero(mask.any(axis=0))
    floor = np.interp(np.arange(mask.shape[1]), valid, bottom[valid]).astype(np.float32)
    sample_y = floor[None, :] + (yy-floor[None, :])/1.65
    extended = cv2.remap(footprint.astype(np.float32), xx, sample_y,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # Fade the new depth rather than making a uniformly dark slab.
    added = .90*extended
    vehicle_height = float(np.ptp(np.where(mask)[0])+1)
    # The original photo may end close to the tires. Fade before that canvas
    # boundary so placement onto a larger background cannot expose a hard cut.
    edge = np.clip((mask.shape[0]-1-yy)/max(2., .12*vehicle_height), 0, 1)
    added *= edge*edge*(3-2*edge)
    added[yy < floor[None, :]] = 0
    added[mask] = 0
    return np.maximum(footprint, added).astype(np.float32)


def choose_shadow(alpha, recovered, expansion=.012):
    """Keep substantial attached source evidence; fill only immediate contact.

    Unlike the production lower-quartile test, this accepts a shadow that is
    intentionally asymmetric or narrow at some points. It does not strengthen
    every pixel or infer a light direction from the vehicle's paint color.
    """
    mask = alpha > 127
    ys, xs = np.where(mask)
    if not len(xs):
        return np.zeros(alpha.shape, np.float32), 'empty'
    bw, bh = np.ptp(xs)+1, np.ptp(ys)+1
    distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    if recovered is not None:
        strong = (recovered > .30) & ~mask & (distance < .18*bh)
        _, columns = np.where(strong)
        # Require distributed evidence, not a lone dark patch near one wheel.
        spread = np.unique(columns).size / bw
        area = strong.sum() / (bw*bh)
        if area > .012 and spread > .35:
            fallback, _ = complete_contact_shadow(alpha, None)
            # A bounded local fill touches the cutout only. Never extend the
            # generic footprint across already recovered source geometry.
            fill = np.minimum(fallback, .50)*np.exp(-.5*(distance/max(1., .012*bh))**2)
            result = np.maximum(recovered, fill)
            # Distributed evidence can still be a very shallow studio shadow.
            # Measure visible attenuation relative to the vehicle's area so a
            # thin strip does not count as a substantial ground footprint.
            mass = float(np.where(mask, 0, np.clip(recovered, 0, 1)).sum() / (bw*bh))
            completion = float(np.clip((.065-mass)/.025, 0, 1))
            if completion > 0:
                result = np.maximum(result, completion*deepen_weak_footprint(alpha, fallback))
            result = expand_ground_shadow(alpha,result,expansion)
            mode = 'source-with-weak-shadow-completion' if completion > 0 else 'preserved-source-with-contact-fill'
            return np.clip(result,0,.97).astype(np.float32), mode
    return complete_contact_shadow(alpha, None)[0], 'synthetic-fallback'


def finish_ground_shadow(alpha, shadow):
    """Round small angular edges and anchor supported lower-contour contacts."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    mask = alpha > 127
    ys, xs = np.where(mask)
    if not len(xs):
        return shadow.copy()
    bw, bh = np.ptp(xs)+1, np.ptp(ys)+1
    radius = max(2, round(.018*min(bw,bh)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    # Fill hidden pixels before opening: otherwise erosion detaches the shadow
    # from the tires. Only the visible ground is affected by the smoothing.
    filled = np.where(mask, .97, shadow).astype(np.float32)
    rounded = cv2.morphologyEx(filled,cv2.MORPH_OPEN,kernel)
    rounded = cv2.GaussianBlur(rounded,(0,0),max(.7,radius*.4))
    distance = cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5)
    blend = np.clip(distance/max(2.,radius*2),0,1)
    result = shadow*(1-blend)+rounded*blend
    yy,xx = np.indices(mask.shape,dtype=np.float32)
    cols = np.arange(xs.min(),xs.max()+1)
    bottom = np.max(np.where(mask,yy,-1),axis=0)[cols]
    lower = gaussian_filter1d(bottom.astype(np.float32),max(1.,.008*bw))
    peaks,_ = find_peaks(lower,prominence=max(2.,.025*bh),distance=max(3,round(.12*bw)))
    peaks = peaks[lower[peaks]>ys.min()+.65*bh]
    for peak in peaks:
        contact_y = bottom[max(0,peak-2):peak+3].max()
        contact = .85*np.exp(-.5*(((xx-cols[peak])/max(2.,.027*bw))**2+
                                 ((yy-contact_y)/max(1.,.009*bh))**2))
        contact[yy<contact_y-1]=0
        result = np.maximum(result,contact)
    result[mask]=shadow[mask]
    return np.clip(result,0,.97).astype(np.float32)


def shadow_from_photo(rgb, alpha, expansion=.012):
    shadow,mode = choose_shadow(alpha, recover_shadow(rgb, alpha), expansion)
    return finish_ground_shadow(alpha,shadow),mode
