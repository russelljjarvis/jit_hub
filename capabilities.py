"""NeuronUnit abstract Capabilities.
The goal is to enumerate all possible capabilities of a model that would be
tested using NeuronUnit. These capabilities exchange 'neo' objects.
"""

import numpy as np
import quantities as pq
import sciunit
import matplotlib.pyplot as plt


class ProducesMembranePotential(sciunit.Capability):
    """Indicates that the model produces a somatic membrane potential."""

    def get_membrane_potential(self, **kwargs):
        """Must return a neo.core.AnalogSignal."""
        raise NotImplementedError()

    def get_mean_vm(self, **kwargs):
        """Get the mean membrane potential."""
        vm = self.get_membrane_potential(**kwargs)
        return np.mean(vm.base)

    def get_median_vm(self, **kwargs):
        """Get the median membrane potential."""
        vm = self.get_membrane_potential(**kwargs)
        return np.median(vm.base)

    def get_std_vm(self, **kwargs):
        """Get the standard deviation of the membrane potential."""
        vm = self.get_membrane_potential(**kwargs)
        return np.std(vm.base)

    def get_iqr_vm(self, **kwargs):
        """Get the inter-quartile range of the membrane potential."""
        vm = self.get_membrane_potential(**kwargs)
        return (np.percentile(vm, 75) - np.percentile(vm, 25))*vm.units

    def get_initial_vm(self, **kwargs):
        """Return a quantity corresponding to the starting membrane potential.
        This will in some cases be the resting potential.
        """
        vm = self.get_membrane_potential(**kwargs)
        return vm[0]  # A neo.core.AnalogSignal object

    def plot_membrane_potential(self, ax=None, ylim=(None, None), **kwargs):
        """Plot the membrane potential."""
        vm = self.get_membrane_potential(**kwargs)
        if ax is None:
            ax = plt.gca()
        vm = vm.rescale('mV')
        ax.plot(vm.times, vm)
        y_min = float(vm.min()-5.0*pq.mV) if ylim[0] is None else ylim[0]
        y_max = float(vm.max()+5.0*pq.mV) if ylim[1] is None else ylim[1]
        ax.set_xlim(vm.times.min(), vm.times.max())
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Vm (mV)')


class ReceivesSquareCurrent(sciunit.Capability):
    """Indicate that somatic current can be injected into the model as
    a square pulse.
    """


class ReceivesCurrent(ReceivesSquareCurrent):
    """Indicate that somatic current can be injected into the model as
    either an arbitrary waveform or as a square pulse.
    """

    def inject_current(self, current):
        """Inject somatic current into the model.
        Parameters
        ----------
        current : neo.core.AnalogSignal
        This is a time series of the current to be injected.
        """
        raise NotImplementedError()
        
    def inject_square_current(self, **kwargs):
        """Injects somatic current into the model.
        Parameters
        ----------
        current : a dictionary like:
                        {'amplitude':-10.0*pq.pA,
                         'delay':100*pq.ms,
                         'duration':500*pq.ms}}
                  where 'pq' is the quantities package
        This describes the current to be injected.
        """
        try:
            self._backend.inject_square_current(**kwargs)
        except AttributeError:
            raise NotImplementedError()
        
    def inject_ramp_current(self, **kwargs):
        """Injects somatic current into the model.
        Parameters
        ----------
        current : a dictionary like:
                        {'amplitude': -10.0*pq.pA,
                         'slope': 1*pq.pA/pq.ms
                         'delay': 100*pq.ms,
                         'duration': 500*pq.ms}}
                  where 'pq' is the quantities package
        This describes the current to be injected.
        """
        raise NotImplementedError()