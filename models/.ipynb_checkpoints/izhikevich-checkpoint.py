from .base import BaseModel
from ..backends.izhikevich import JIT_IzhiBackend
from copy import copy


class IzhiModel(BaseModel):
    def __init__(self, name=None, attrs=None, backend=JIT_IzhiBackend):
        self.default_attrs = {'C':89.7960714285714,
                              'a':0.01, 'b':15, 'c':-60, 'd':10, 'k':1.6,
                              'vPeak':(86.364525297619-65.2261863636364),
                              'vr':-65.2261863636364, 'vt':-50, 'celltype':3}
        if attrs is None:
            attrs = {}
        attrs_ = copy(self.default_attrs)
        for key, value in attrs:
            attrs_[key] = value
        super().__init__(name=name, attrs=attrs_, backend=backend)
        

