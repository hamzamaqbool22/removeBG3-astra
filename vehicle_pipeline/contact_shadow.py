"""Conservative ground contact completion; never projects the full car mask."""
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


def complete_contact_shadow(alpha, recovered):
    """Keep substantial photographed shadows; replace weak grounding with a synthesized ground footprint.

    The lower contour supplies position only. Width/height and opacity are
    synthesized independently, with no extrapolation beside vertical body edges.
    Returns (shadow, added_layer) for inspection and regression checks.
    """
    mask = alpha > 127
    ys, xs = np.where(mask)
    empty = np.zeros(alpha.shape, np.float32)
    if not len(xs):
        return (empty if recovered is None else recovered), empty
    x0,x1,y0,y1 = xs.min(),xs.max(),ys.min(),ys.max()
    bw,bh = x1-x0+1,y1-y0+1
    if min(bw,bh) < 20:
        return (empty if recovered is None else recovered), empty
    columns = np.arange(x0,x1+1)
    bottom = mask.shape[0]-1-np.argmax(mask[::-1,columns],axis=0)
    valid = mask[:,columns].any(axis=0) & (bottom >= y0+.70*bh)
    # Assess the lower quartile across the central ground region, not a peak
    # under one tire. A small surviving patch is not complete grounding.
    sample_y = bottom + max(1,round(.06*bh))
    samples = valid & (columns>x0+.10*bw) & (columns<x1-.10*bw) & (sample_y<mask.shape[0])
    if recovered is not None:
        if samples.sum() < 20:
            return recovered, empty
        coverage = float(np.quantile(recovered[sample_y[samples],columns[samples]],.25))
        near_y = np.minimum(mask.shape[0]-1,bottom + max(1,round(.008*bh)))
        near_strength = float(np.quantile(recovered[near_y[samples],columns[samples]],.25))
        # Side-on photographs can have a thin but already convincing dark
        # contact strip. Do not require a wide cast shadow in that case.
        if coverage >= .35 or near_strength >= .84:
            return recovered, empty
    # Fit a simple ground footprint to the central lower body. Do not trace
    # every exhaust/tire notch: those made the previous band look like an outline.
    u = (columns-(x0+x1)/2)/max(1.,bw/2)
    fit = valid & (np.abs(u)<.82)
    if fit.sum()<10:
        return (empty if recovered is None else recovered), empty
    lower = gaussian_filter1d(median_filter(bottom.astype(np.float32),size=max(3,round(.025*bw)|1)),max(1,.025*bw))
    coeff = np.polyfit(u[fit],lower[fit],2)
    curve = np.polyval(coeff,u)
    # Limit extrapolation; an uncertain end of the contour cannot lift the
    # receiving ground plane alongside the body.
    curve = np.clip(curve,y0+.70*bh,y1+.02*bh)
    yy = np.arange(mask.shape[0],dtype=np.float32)[:,None]
    dy = yy-curve[None,:]
    end_view = bw/bh < 1.65
    depth = bh*(.090 if end_view else .045)
    depth *= .65+.35*np.sqrt(np.clip(1-u*u,0,1))
    lateral = np.exp(-2.3*np.abs(u)**6)
    edge = np.clip((1-np.abs(u))/.13,0,1)
    lateral *= edge*edge*(3-2*edge)*valid
    # A full underbody ground field, darkest near the vehicle and softer and
    # lighter farther out. These are generated densities, never warped alpha.
    core = .88*np.exp(-.5*((dy+.008*bh)/np.maximum(1,depth)[None,:])**2)
    outer = .14*np.exp(-.5*(dy/np.maximum(1,depth*1.65)[None,:])**2)
    contact = .95*np.exp(-.5*((yy-lower[None,:])/max(1,.012*bh))**2)
    band = np.maximum(np.maximum(core,outer),contact)*lateral[None,:]
    # Smoothly bound the field near the actual lower edge to avoid a vertical
    # artifact at bumper corners. Opaque car pixels are never modified.
    top = np.clip((yy-bottom[None,:]+.01*bh)/max(1,.01*bh),0,1)
    band *= top*top*(3-2*top)
    far = np.clip((.38*bh-dy)/max(1,.08*bh),0,1)
    band *= far*far*(3-2*far)
    added = empty.copy()
    added[:,columns] = band
    added[mask] = 0
    # Weak source shadows contain floor-pattern fragments and uncertain edges.
    # Replace them instead of retaining those fragments through max-compositing.
    return np.clip(added,0,.97).astype(np.float32), added
