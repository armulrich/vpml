"""Fixed parameter coordinates for the unchanged full-family scalar objective."""
import numpy as np


def initial_metric(params, mode):
    if mode == 'euclidean':
        return None
    if mode != 'initial_rms':
        raise ValueError(mode)
    metric = {}
    for key,value in params.items():
        a = np.asarray(value)
        squared_rms = float(np.mean(a*a))
        if squared_rms <= 0 or not np.isfinite(squared_rms):
            raise ValueError(f'Initial RMS must be positive: {key}')
        metric[key] = (squared_rms,float(np.linalg.norm(a)))
    return metric


def proposal(gradient, loss, metric):
    """Return direction, step and -g dot direction for Armijo acceptance.

    The RMS metric is a fixed diagonal change of parameter coordinates, not
    adaptive loss reweighting. Bound each tensor's proposed displacement to
    10% of its initial norm, with a 1% predicted scalar-loss decrease.
    """
    if metric is None:
        norm = np.sqrt(sum(np.sum(v*v) for v in gradient.values()))
        if norm <= 0 or not np.isfinite(norm):
            raise ValueError('No finite descent gradient')
        return {k:-v/norm for k,v in gradient.items()},min(1e-4,.01*loss/norm),norm
    direction = {k:-metric[k][0]*v for k,v in gradient.items()}
    slope = -sum(float(np.sum(gradient[k]*v)) for k,v in direction.items())
    relative = max(float(np.linalg.norm(v))/metric[k][1] for k,v in direction.items())
    if slope <= 0 or relative <= 0 or not np.isfinite(slope+relative):
        raise ValueError('No finite metric descent direction')
    return direction,min(.01*loss/slope,.1/relative),slope
