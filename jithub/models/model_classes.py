from .base import BaseModel
from .backends.izhikevich import JIT_IZHIBackend
from .backends.mat_nu import JIT_MATBackend
from .backends.adexp import JIT_ADEXPBackend

from copy import copy
import collections
import quantities as pq
from sciunit import capabilities as scap
from neuronunit import capabilities as ncap

from bluepyopt.ephys.models import CellModel
class BPOModel(CellModel,ncap.ProducesMembranePotential,scap.Runnable):
    def __init__(self):
        self.name = "neuronunit_numba_model"
        super(BPOModel, self).__init__(self.name)
        """
        TODO
        # Force garbage collection of NEURON/HOC specific code
        # in case its resource intensive.
        #del self.create_hoc
        #del self.create_empty_cell
        #del self.destroy
        #del self.instantiate
        """
    def get_AP_widths(self):
        from neuronunit.capabilities import spike_functions as sf
        vm = self.get_membrane_potential()
        widths = sf.spikes2widths(vm)
        return widths

    def inject_model(self,
                     DELAY=1000.0*pq.ms,
                     DURATION=2000.0*pq.ms):
        from neuronunit.optimization import optimization_management
        dynamic_attrs = {str(k):float(v) for k,v in self.params.items()}
        frozen_attrs = self.default_attrs
        frozen_attrs.update(dynamic_attrs)
        all_attrs = frozen_attrs
        dtc = self.model_to_dtc(attrs=all_attrs)
        assert len(dtc.attrs)
        dtc = dtc_to_rheo(dtc)
        self.rheobase = dtc.rheobase
        vm = [np.nan]
        if self.rheobase is not None:
            uc = {'amplitude':self.rheobase,'duration':DURATION,'delay':DELAY}
            self._backend.attrs = all_attrs
            try:
                self._backend.inject_square_current(uc)
                vm = self.get_membrane_potential()
                self.vm = vm
            except:
                vm = [np.nan]
        else:
            self.vm = vm
        return vm

    def freeze(self, param_dict):
        """
        Over ride parent class method
        Set params

        """

        for param_name, param_value in param_dict.items():
            if hasattr(self.params[param_name],'freeze'):# is type(np.float):
                self.params[param_name].freeze(param_value)
            else:
                from bluepyopt.parameters import Parameter

                self.params[param_name] = Parameter(name=param_name,value=param_value,frozen=True)


    def instantiate(self, sim=None):
        """
        Over ride parent class method
        Instantiate model in simulator
        As if called from a genetic algorithm.
        """
        if self.params is not None:
            self.attrs = self.params

        dtc = self.model_to_dtc()
        for k,v in self.params.items():
            if hasattr(v,'value'):
                v = float(v.value)

            dtc.attrs[k] = v
            self.attrs[k] = v
        return dtc


    def model_to_dtc(self,attrs=None):
        """
        Args:
            self
        Returns:
            dtc
            DTC is a simulator indipendent data transport container object.
        """
        from neuronunit.optimization.data_transport_container import DataTC

        dtc = DataTC(backend=self.backend)
        dtc.attrs = self.attrs
        return dtc

        if type(attrs) is not type(None):
            if len(attrs):
                dtc.attrs = attrs
                self.attrs = attrs
            assert self._backend is not None
            return dtc
        else:
            if type(self.attrs) is not type(None):
                if len(self.attrs):
                    try:
                        dynamic_attrs = {str(k):float(v) for k,v in self.attrs.items()}
                    except:
                        dynamic_attrs = {str(k):float(v.value) for k,v in self.attrs.items()}

        if self._backend is None:
            super(VeryReducedModel, self).__init__(name=self.name,backend=self.backend)#,attrs=dtc.attrs)
            assert self._backend is not None
        frozen_attrs = self._backend.default_attrs
        if 'dynamic_attrs' in locals():
            frozen_attrs.update(dynamic_attrs)
        all_attrs = frozen_attrs
        dtc.attrs = all_attrs
        assert dtc.attrs is not None
        return dtc


    def check_nonfrozen_params(self, param_names):
        """
        Over ride parent class method
        Check if all nonfrozen params are set"""
        for param_name, param in self.params.items():
            if not param.frozen:
                raise Exception(
                    'CellModel: Nonfrozen param %s needs to be '
                    'set before simulation' %
                    param_name)



class ADEXPModel(BaseModel,BPOModel):
    def __init__(self, name=None, params=None, backend=JIT_ADEXPBackend):
        self.default_attrs = {}
        self.default_attrs['cm']=0.281
        self.default_attrs['v_spike']=-40.0
        self.default_attrs['v_reset']=-70.6
        self.default_attrs['v_rest']=-70.6
        self.default_attrs['tau_m']=9.3667
        self.default_attrs['a']=4.0
        self.default_attrs['b']=0.0805
        self.default_attrs['delta_T']=2.0
        self.default_attrs['tau_w']=144.0
        self.default_attrs['v_thresh']=-50.4
        self.default_attrs['spike_delta']=30

        if params is not None:
            self.params = collections.OrderedDict(**params)
        else:
            self.params = self.default_attrs
        super().__init__(name=name, attrs=self.params, backend=backend)


class IzhiModel(BaseModel,BPOModel):
    def __init__(self, name=None, params=None, backend=JIT_IZHIBackend):
        self.default_attrs = {'C':89.7960714285714,
                              'a':0.01, 'b':15, 'c':-60, 'd':10, 'k':1.6,
                              'vPeak':(86.364525297619-65.2261863636364),
                              'vr':-65.2261863636364, 'vt':-50, 'celltype':3}
        if params is not None:
            self.params = collections.OrderedDict(**params)
        else:
            self.params = self.default_attrs
        super().__init__(name=name, attrs=self.params, backend=backend)



class MATModel(BaseModel,BPOModel):
    def __init__(self, name=None, attrs=None, backend=JIT_MATBackend):
        self.default_attrs = {'vr':-65.0,'vt':-55.0,'a1':10, 'a2':2, 'b':0, 'w':5, 'R':10, 'tm':10, 't1':10, 't2':200, 'tv':5, 'tref':2}
        if attrs is None:
            attrs = {}
        attrs_ = copy(self.default_attrs)
        for key, value in attrs:
            attrs_[key] = value
        super().__init__(name=name, attrs=attrs_, backend=backend)
