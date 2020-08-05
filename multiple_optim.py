import torch.optim as optim


class MultipleOptimizer(object):
    def __init__(self, op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

    def return_optimizers(self):
        return self.optimizers


def make_muliti_optim(parameters):
    """build compound optimizer for adam and sparseadam"""

    params = []
    sparse_params = []
    for k, p in parameters:
        if p.requires_grad:
            if "embed" not in k:
                params.append(p)
            else:
                sparse_params.append(p)
    optimizer = MultipleOptimizer(
        [optim.Adam(params, lr=1.0e-3,), optim.SparseAdam(sparse_params, lr=1.0e-3)]
    )
    return optimizer
