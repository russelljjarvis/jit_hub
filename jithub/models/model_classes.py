from .base import BaseModel
from jithub.backends.izhikevich import JIT_IzhiBackend
from jithub.backends.mat_nu import JIT_MATBackend
from jithub.backends.adexp import JIT_ADEXPBackend

from copy import copy
import collections


class BPOModel():
    #def get_membrane_potential(self):
    #    self._backend.get_membrane_potential()

    def check_name(self):
        """Check if name complies with requirements"""

        allowed_chars = string.ascii_letters + string.digits + '_'

        if sys.version_info[0] < 3:
            translate_args = [None, allowed_chars]
        else:
            translate_args = [str.maketrans('', '', allowed_chars)]

        if self.name == '' \
                or self.name[0] not in string.ascii_letters \
                or not str(self.name).translate(*translate_args) == '':
            raise TypeError(
                'CellModel: name "%s" provided to constructor does not comply '
                'with the rules for Neuron template name: name should be '
                'alphanumeric '
                'non-empty string, underscores are allowed, '
                'first char should be letter' % self.name)

    def params_by_names(self, param_names):
        """Get parameter objects by name"""

        return [self.params[param_name] for param_name in param_names]

    def freeze(self, param_dict):
        """Set params"""

        for param_name, param_value in param_dict.items():
            if hasattr(self.params[param_name],'freeze'):# is type(np.float):
                self.params[param_name].freeze(param_value)
            else:
                from bluepyopt.parameters import Parameter

                self.params[param_name] = Parameter(name=param_name,value=param_value,frozen=True)
                #self.params[param_name].freeze(param_value)
                #self.params[param_name] = param_value


    def unfreeze(self, param_names):
        """Unset params"""

        for param_name in param_names:
            self.params[param_name].unfreeze()

    def instantiate(self, sim=None):
        """
        Instantiate model in simulator
        As if called from a genetic algorithm.
        """
        #self.icell.gid = self.gid
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
        from neuronunit.optimisation.data_transport_container import DataTC

        dtc = DataTC(backend=self.backend)
        dtc.attrs = self.attrs
        return dtc
        #if hasattr(self,'tests'):
        #    if type(self.tests) is not type(None):
        #        dtc.tests = self.tests

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

            #for param in self.params.values():
            #    model.attrs[param] =
                #param.instantiate(sim=sim, icell=self.icell)
    def destroy(self, sim=None):  # pylint: disable=W0613
        """Destroy instantiated model in simulator"""

        # Make sure the icell's destroy() method is called
        # without it a circular reference exists between CellRef and the object
        # this prevents the icells from being garbage collected, and
        # cell objects pile up in the simulator
        self.icell.destroy()

        # The line below is some M. Hines magic
        # DON'T remove it, because it will make sure garbage collection

        del self.icell# = None
        for param in self.params.values():
            param.destroy(sim=sim)
            print('destroyed param')

    def check_nonfrozen_params(self, param_names):  # pylint: disable=W0613
        """Check if all nonfrozen params are set"""
        for param_name, param in self.params.items():
            if not param.frozen:
                raise Exception(
                    'CellModel: Nonfrozen param %s needs to be '
                    'set before simulation' %
                    param_name)



class ADEXPModel(BaseModel,BPOModel):
    def __init__(self, name=None, params=None, backend=JIT_ADEXPBackend):
        BAE1 = {}
        BAE1['cm']=0.281
        BAE1['v_spike']=-40.0
        BAE1['v_reset']=-70.6
        BAE1['v_rest']=-70.6
        BAE1['tau_m']=9.3667
        BAE1['a']=4.0
        BAE1['b']=0.0805

        BAE1['delta_T']=2.0
        BAE1['tau_w']=144.0
        BAE1['v_thresh']=-50.4
        BAE1['spike_delta']=30

        self.default_attrs = BAE1
        if params is not None:
            self.params = collections.OrderedDict(**params)
        else:
            self.params = self.default_attrs
        super().__init__(name=name, attrs=self.params, backend=backend)
    #def get_membrane_potential(self):
        #super().__init__(name=name, attrs=self.params, backend=backend)
    #    self._backend.get_membrane_potential()
        #print('gets here')

class IzhiModel(BaseModel,BPOModel):
    def __init__(self, name=None, params=None, backend=JIT_IzhiBackend):
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
