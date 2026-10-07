"""Conservative ground contact completion; never projects the full car mask."""
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter
from scipy.signal import find_peaks


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
    # Smooth the lower contour so perspective-raised rear tires retain contact.
    # A global quadratic and hard height gate can cut the shadow off mid-body.
    u = (columns-(x0+x1)/2)/max(1.,bw/2)
    fit = valid & (np.abs(u)<.82)
    if fit.sum()<10:
        return (empty if recovered is None else recovered), empty
    lower = gaussian_filter1d(median_filter(bottom.astype(np.float32),size=max(3,round(.025*bw)|1)),max(1,.025*bw))
    curve = lower.copy()
    # Tires protrude below the rocker panels. Following every contour valley
    # produces a thin halo instead of a ground footprint, especially on SUVs.
    # Bridge supported contact peaks in image coordinates, retaining perspective
    # (including a third visible wheel), without assuming a vehicle/view label.
    contacts, _ = find_peaks(lower, prominence=max(2., .025*bh),
                             distance=max(3, round(.12*bw)))
    contacts = contacts[lower[contacts] >= y0+.50*bh]
    supported_wheels = len(contacts) >= 2 and contacts[-1]-contacts[0] >= .25*bw
    if supported_wheels:
        ground = np.interp(np.arange(bw), contacts, lower[contacts])
        # Continue the outer tire's ground height beneath its bumper overhang.
        # Following the bumper upward makes the shadow disappear at either end.
        # Constant extension avoids amplifying an uncertain perspective slope.
        curve = np.maximum(curve, ground)
        curve = gaussian_filter1d(curve, max(1., .012*bw))
    # Limit extrapolation; an uncertain end of the contour cannot lift the
    # receiving ground plane alongside the body.
    curve = np.clip(curve,y0+.45*bh,y1+.02*bh)
    yy = np.arange(mask.shape[0],dtype=np.float32)[:,None]
    dy = yy-curve[None,:]
    end_view = bw/bh < 1.65
    depth = bh*(.090 if end_view else .060 if supported_wheels else .055)
    depth *= .65+.35*np.sqrt(np.clip(1-u*u,0,1))
    lateral = np.exp(-.8*np.abs(u)**6)
    edge = np.clip((1-np.abs(u))/(.045 if supported_wheels else .13),0,1)
    support = np.clip((lower-(y0+.45*bh))/max(1,.12*bh),0,1)
    lateral *= edge*edge*(3-2*edge)*support*support*(3-2*support)
    # A full underbody ground field, darkest near the vehicle and softer and
    # lighter farther out. These are generated densities, never warped alpha.
    # The visible space between rocker and contact plane is occluded ground,
    # not an illuminated hole. Falloff starts beyond that plane; the silhouette
    # gate below still prevents painting shadows up vertical body edges.
    core = (.88 if end_view else .76 if supported_wheels else .66)*np.exp(-.5*(np.maximum(dy+.008*bh,0)/np.maximum(1,depth)[None,:])**2)
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
    # Feather in both axes before occluding with the car. Column-wise gates
    # otherwise leave straight seams near steep bumper and tire transitions.
    added = gaussian_filter(added, sigma=max(.7,.003*bh))
    if supported_wheels:
        # Use a narrow, low-density feather around the footprint. Wide
        # diffusion or independent bumper lobes create detached dark patches
        # beside the car instead of a connected underbody shadow.
        diffuse = gaussian_filter(added, sigma=(max(1., .020*bh), max(1., .012*bw)))
        added = np.maximum(added, .60*diffuse)
    else:
        added[:,:x0] = 0
        added[:,x1+1:] = 0
    if supported_wheels:
        # The broad contour is deliberately smoothed, but tire contact must
        # meet the actual rubber edge. Add small, crisp ground patches after
        # diffusion so the blur cannot lift the tires off the pavement.
        xx = np.arange(mask.shape[1], dtype=np.float32)[None, :]
        radius = max(2, round(.045*bw))
        for peak in contacts:
            start, stop = max(0, peak-radius), min(bw, peak+radius+1)
            local_bottom = bottom[start:stop]
            contact_y = float(np.quantile(local_bottom, .98))
            sole = np.flatnonzero(local_bottom >= contact_y-1)
            contact_x = x0 + start + float(np.mean(sole))
            width = max(3., min(.045*bw, max(.028*bw, len(sole)*.75)))
            tight = .96*np.exp(-.5*(((xx-contact_x)/width)**2 +
                                   ((yy-contact_y-.003*bh)/max(1., .010*bh))**2))
            soft = .65*np.exp(-.5*(((xx-contact_x)/(width*1.6))**2 +
                                   ((yy-contact_y)/max(2., .024*bh))**2))
            added = np.maximum(added, np.maximum(tight, soft))
    # Source photos can end inside the penumbra. Fade at the source canvas
    # boundary so placement on a larger background cannot reveal a rectangle.
    h, w = mask.shape
    x_distance = np.minimum(np.arange(w), np.arange(w)[::-1])
    y_distance = np.minimum(np.arange(h), np.arange(h)[::-1])
    edge_x = np.clip(x_distance / max(2., .06*bw), 0, 1)
    edge_y = np.clip(y_distance / max(2., .10*bh), 0, 1)
    added *= (edge_y*edge_y*(3-2*edge_y))[:, None]
    added *= (edge_x*edge_x*(3-2*edge_x))[None, :]
    added[mask] = 0
    # Weak source shadows contain floor-pattern fragments and uncertain edges.
    # Replace them instead of retaining those fragments through max-compositing.
    return np.clip(added,0,.97).astype(np.float32), added
