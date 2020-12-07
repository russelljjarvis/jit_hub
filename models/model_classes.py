from .base import BaseModel
from backends.izhikevich import JIT_IzhiBackend
from backends.mat_nu import JIT_MATBackend

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




class MATModel(BaseModel):
    def __init__(self, name=None, attrs=None, backend=JIT_MATBackend):
        self.default_attrs = {'vr':-65.0,'vt':-55.0,'a1':10, 'a2':2, 'b':0, 'w':5, 'R':10, 'tm':10, 't1':10, 't2':200, 'tv':5, 'tref':2}
        if attrs is None:
            attrs = {}
        attrs_ = copy(self.default_attrs)
        for key, value in attrs:
            attrs_[key] = value
        super().__init__(name=name, attrs=attrs_, backend=backend)
