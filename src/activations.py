"""
This code has been copied and modified from [Sparse Feature Circuits](https://github.com/saprmarks/feature-circuits).
"""

from __future__ import annotations
import torch as t
from torchtyping import TensorType



class SparseActivation():
    """
    A SparseActivation is a helper class which represents a vector in the sparse feature basis provided by an SAE, jointly with the SAE error term.
    A SparseActivation may have three fields:
    act : the feature activations in the sparse basis
    res : the SAE error term
    resc : a contracted SAE error term, useful for when we want one number per feature and error (instead of having d_model numbers per error)
    """

    def __init__(
            self, 
            act: TensorType["batch_size", "n_ctx", "d_dictionary"] = None, 
            res: TensorType["batch_size", "n_ctx", "d_model"] = None,
            resc: TensorType["batch_size", "n_ctx"] = None, # contracted residual
            ) -> None:

            self.act = act
            self.res = res
            self.resc = resc

    def _map(self, f, aux=None) -> 'SparseActivation':
        kwargs = {}
        if isinstance(aux, SparseActivation):
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None and getattr(aux, attr) is not None:
                    kwargs[attr] = f(getattr(self, attr), getattr(aux, attr))
        else:
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = f(getattr(self, attr), aux)
        return SparseActivation(**kwargs)
        
    def __mul__(self, other) -> 'SparseActivation':
        if isinstance(other, SparseActivation):
            # Handle SparseActivation * SparseActivation
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) * getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) * other
        return SparseActivation(**kwargs)

    def __rmul__(self, other) -> 'SparseActivation':
        # This will handle float/int * SparseActivation by reusing the __mul__ logic
        return self.__mul__(other)
    
    def __matmul__(self, other: SparseActivation) -> SparseActivation:
        # dot product between two SparseActivations, except only the residual is contracted
        return SparseActivation(act = self.act * other.act, resc=(self.res * other.res).sum(dim=-1, keepdim=True))
    
    def __add__(self, other) -> 'SparseActivation':
        if isinstance(other, SparseActivation):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    self_attr = getattr(self, attr)
                    other_attr = getattr(other, attr)
                    if self_attr.shape != other_attr.shape:
                        if self_attr.shape[0] < other_attr.shape[0]:
                            self_attr = t.nn.functional.pad(self_attr, (0, 0, 0, other_attr.shape[0] - self_attr.shape[0]))
                        else:
                            other_attr = t.nn.functional.pad(other_attr, (0, 0, 0, self_attr.shape[0] - other_attr.shape[0]))
                    kwargs[attr] = self_attr + other_attr
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) + other
        return SparseActivation(**kwargs)
    
    def __radd__(self, other: SparseActivation) -> SparseActivation:
        return self.__add__(other)
    
    def __sub__(self, other: SparseActivation) -> SparseActivation:
        if isinstance(other, SparseActivation):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    if getattr(self, attr).shape != getattr(other, attr).shape:
                        raise ValueError(f"Shapes of {attr} do not match: {getattr(self, attr).shape} and {getattr(other, attr).shape}")
                    kwargs[attr] = getattr(self, attr) - getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) - other
        return SparseActivation(**kwargs)
    
    def __truediv__(self, other) -> SparseActivation:
        if isinstance(other, SparseActivation):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) / getattr(other, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) / other
        return SparseActivation(**kwargs)

    def __rtruediv__(self, other) -> SparseActivation:
        if isinstance(other, SparseActivation):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = other / getattr(self, attr)
        else:
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = other / getattr(self, attr)
        return SparseActivation(**kwargs)

    def __neg__(self) -> SparseActivation:
        sparse_result = -self.act
        res_result = -self.res
        return SparseActivation(act=sparse_result, res=res_result)
    
    def __invert__(self) -> SparseActivation:
            return self._map(lambda x, _: ~x)


    def __gt__(self, other) -> SparseActivation:
        if isinstance(other, (int, float)):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) > other
            return SparseActivation(**kwargs)
        raise ValueError("SparseActivation can only be compared to a scalar.")
    
    def __lt__(self, other) -> SparseActivation:
        if isinstance(other, (int, float)):
            kwargs = {}
            for attr in ['act', 'res', 'resc']:
                if getattr(self, attr) is not None:
                    kwargs[attr] = getattr(self, attr) < other
            return SparseActivation(**kwargs)
        raise ValueError("SparseActivation can only be compared to a scalar.")
    
    def __getitem__(self, index: int):
        return self.act[index]
    
    def __repr__(self):
        if self.res is None:
            return f"SparseActivation(act={self.act}, resc={self.resc})"
        if self.resc is None:
            return f"SparseActivation(act={self.act}, res={self.res})"
        else:
            raise ValueError("SparseActivation has both residual and contracted residual. This is an unsupported state.")
    
    def sum(self, dim=None):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).sum(dim)
        return SparseActivation(**kwargs)
    
    def mean(self, dim: int):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).mean(dim)
        return SparseActivation(**kwargs)
    
    def nonzero(self):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).nonzero()
        return SparseActivation(**kwargs)
    
    def squeeze(self, dim: int):
        kwargs = {}
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                kwargs[attr] = getattr(self, attr).squeeze(dim)
        return SparseActivation(**kwargs)

    @property
    def grad(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).grad
        return SparseActivation(**kwargs)
    
    def clone(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).clone()
        return SparseActivation(**kwargs)
    
    @property
    def value(self):
        kwargs = {}
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                kwargs[attribute] = getattr(self, attribute).value
        return SparseActivation(**kwargs)

    def save(self):
        for attribute in ['act', 'res', 'resc']:
            if getattr(self, attribute) is not None:
                setattr(self, attribute, getattr(self, attribute).save())
        return self
    
    def detach(self):
        self.act = self.act.detach()
        self.res = self.res.detach()
        return SparseActivation(act=self.act, res=self.res)
    
    def to_tensor(self):
        if self.resc is None:
            return t.cat([self.act, self.res], dim=-1)
        if self.res is None:
            return t.cat([self.act, self.resc], dim=-1)
        raise ValueError("SparseActivation has both residual and contracted residual. This is an unsupported state.")

    def to(self, device):
        for attr in ['act', 'res', 'resc']:
            if getattr(self, attr) is not None:
                setattr(self, attr, getattr(self, attr).to(device))
        return self
    
    def __gt__(self, other):
        return self._map(lambda x, y: x > y, other)
    
    def __lt__(self, other):
        return self._map(lambda x, y: x < y, other)
    
    def nonzero(self):
        return self._map(lambda x, _: x.nonzero())
    
    def squeeze(self, dim):
        return self._map(lambda x, _: x.squeeze(dim=dim))
    
    def expand_as(self, other):
        return self._map(lambda x, y: x.expand_as(y), other)
    
    def zeros_like(self):
        return self._map(lambda x, _: t.zeros_like(x))
    
    def ones_like(self):
        return self._map(lambda x, _: t.ones_like(x))
    
    def abs(self):
        return self._map(lambda x, _: x.abs())
