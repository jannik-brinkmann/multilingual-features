"""
This code has been copied and modified from [Sparse Feature Circuits](https://github.com/saprmarks/feature-circuits).
"""

import torch

from .activations import SparseActivation


TRACER_KWARGS = {
    'scan' : False,
    'validate' : False
}


def attribution_patching(
    clean_prefix,
    patch_prefix,
    clean_answer,
    patch_answer,
    model,
    submodules,
    dictionaries,
    steps=10,
    metric_kwargs=dict(),
):

    clean_prefix = torch.cat([clean_prefix], dim=0).to("cuda")
    patch_prefix = torch.cat([patch_prefix], dim=0).to("cuda")

    def metric_fn(model):
        return (
            torch.gather(model.lm_head.output[:,-1,:], dim=-1, index=torch.tensor([patch_answer.item()]).view(-1, 1)).squeeze(-1) - \
            torch.gather(model.lm_head.output[:,-1,:], dim=-1, index=torch.tensor([clean_answer.item()]).view(-1, 1)).squeeze(-1)
        )
    
    # first run through a test input to figure out which hidden states are tuples
    is_tuple = {}
    with model.trace("_", **TRACER_KWARGS):
        for submodule in submodules:
            is_tuple[submodule] = True # type(submodule.output) == tuple

    hidden_states_clean = {}
    with model.trace(clean_prefix, **TRACER_KWARGS), torch.no_grad():
        for submodule in submodules:
            dictionary = dictionaries[submodule]
            x = submodule.output
            if is_tuple[submodule]:
                x = x[0]
            f = dictionary.encode(x)
            x_hat = dictionary.decode(f)
            residual = x - x_hat
            hidden_states_clean[submodule] = SparseActivation(act=f.save(), res=residual.save())
        metric_clean = metric_fn(model, **metric_kwargs).save()
    hidden_states_clean = {k : v.value for k, v in hidden_states_clean.items()}

    if patch_prefix is None:
        hidden_states_patch = {
            k : SparseActivation(act=torch.zeros_like(v.act), res=torch.zeros_like(v.res)) for k, v in hidden_states_clean.items()
        }
        total_effect = None
    else:
        hidden_states_patch = {}
        with model.trace(patch_prefix, **TRACER_KWARGS), torch.no_grad():
            for submodule in submodules:
                dictionary = dictionaries[submodule]
                x = submodule.output
                if is_tuple[submodule]:
                    x = x[0]
                f = dictionary.encode(x)
                x_hat = dictionary.decode(f)
                residual = x - x_hat
                hidden_states_patch[submodule] = SparseActivation(act=f.save(), res=residual.save())
            metric_patch = metric_fn(model, **metric_kwargs).save()
        total_effect = (metric_patch.value - metric_clean.value).detach()
        hidden_states_patch = {k : v.value for k, v in hidden_states_patch.items()}

    effects = {}
    deltas = {}
    grads = {}
    for submodule in submodules:
        dictionary = dictionaries[submodule]
        clean_state = hidden_states_clean[submodule]
        patch_state = hidden_states_patch[submodule]
        with model.trace(**TRACER_KWARGS) as tracer:
            metrics = []
            fs = []
            for step in range(steps):
                alpha = step / steps
                f = (1 - alpha) * clean_state + alpha * patch_state
                f.act.retain_grad()
                f.res.retain_grad()
                fs.append(f)
                with tracer.invoke(clean_prefix, scan=TRACER_KWARGS['scan']):
                    if is_tuple[submodule]:
                        decoded = dictionary.decode(f.act.to(dictionary.device))
                        submodule.output[0][:] = decoded.to(model.device) + f.res
                    else:
                        decoded = dictionary.decode(f.act.to(dictionary.device))
                        submodule.output = decoded.to(model.device) + f.res
                    metrics.append(metric_fn(model, **metric_kwargs))
            metric = sum([m for m in metrics])
            metric.sum().backward(retain_graph=True) # TODO : why is this necessary? Probably shouldn't be, contact jaden

        mean_grad = sum([f.act.grad for f in fs]) / steps
        mean_residual_grad = sum([f.res.grad for f in fs]) / steps
        grad = SparseActivation(act=mean_grad, res=mean_residual_grad)
        delta = (patch_state - clean_state).detach() if patch_state is not None else -clean_state.detach()
        effect = grad @ delta

        effects[submodule] = effect
        deltas[submodule] = delta
        grads[submodule] = grad

    return (effects, deltas, grads, total_effect)
