"""Balance expert/shared descent contributions to the unchanged scalar loss."""
import numpy as np
from model.train.parameter_metric import proposal


def balanced_proposal(gradient, loss, metric):
    if metric is None:
        raise ValueError('Expert balancing requires an initial RMS metric')
    is_expert = lambda key: key.startswith(('expert_', 'specialist_'))
    shared = sum(metric[k][0]*float(np.sum(v*v)) for k,v in gradient.items()
                 if not is_expert(k))
    experts = sum(metric[k][0]*float(np.sum(v*v)) for k,v in gradient.items()
                  if is_expert(k))
    if min(shared, experts) <= 0 or not np.isfinite(shared+experts):
        raise ValueError('Both groups require finite positive descent')
    multiplier = shared/experts
    adjusted = {k:(scale*(multiplier if is_expert(k) else 1.),norm)
                for k,(scale,norm) in metric.items()}
    return proposal(gradient, loss, adjusted)
