import sciunit
from sciunit.models.runnable import RunnableModel
from sciunit.models.backends import Backend
from quantities import mV, ms, s, V
import quantities as qt
from neo import AnalogSignal
import numpy as np
import quantities as pq
import numpy
import copy
from elephant.spike_train_generation import threshold_detection
from capabilities import ProducesMembranePotential, ReceivesCurrent


class BaseModel(RunnableModel, ProducesMembranePotential, ReceivesCurrent):
    name = None

    def __init__(self, name=None, attrs=None, backend=None):
        super().__init__(name=name, attrs=attrs, backend=backend)
        self.vM = None
        self.attrs = attrs
        self.temp_attrs = None
        self.default_attrs = {}

        if type(attrs) is not type(None):
            self.attrs = attrs
        if self.attrs is None:
            self.attrs = self.default_attrs


    def get_spike_count(self):
        self.vM = self._backend.get_membrane_potential()
        thresh = threshold_detection(self.vM,0*qt.mV)
        ##
        # Reasoning for condition:
        # humans think spikes have down strokes in addition
        # to the upstroke, but only upstrokes are
        # detected by threshold_detection
        ##
        if self.vM[-1]>0:
            spikes = len(thresh) -1
        else:
            spikes = len(thresh)
        return spikes


    def set_stop_time(self, stop_time = 650*pq.ms):
        """Sets the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.tstop = float(stop_time.rescale(pq.ms))


    def get_membrane_potential(self):
        self.vM = self._backend.get_membrane_potential()
        return self.vM


    def set_attrs(self, attrs):
        self.attrs = attrs
        self._backend.attrs = attrs

    '''
    def step(amplitude, t_stop):
       times = np.array([0, t_stop/10, t_stop])
       amps = np.array([0, amplitude, amplitude])
       return times, amps


    def pulse(amplitude, onsets, width, t_stop, baseline=0.0):
        times = [0]
        amps = [baseline]
        for onset in onsets:
           times += [onset, onset + width]
           amps += [amplitude, baseline]
        times += [t_stop]
        amps += [baseline]
        return np.array(times), np.array(amps)


    def ramp(self,gradient, onset, t_stop, baseline=0.0, time_step=0.125, t_start=0.0):
        if onset > t_start:
            times = np.hstack((np.array((t_start, onset)),  # flat part
                    np.arange(onset + time_step, t_stop + time_step, time_step)))  # ramp part
        else:
            times = np.arange(t_start, t_stop + time_step, time_step)
        amps = baseline + gradient*(times - onset) * (times > onset)
        return times, amps


    def stepify(times, values):
        new_times = np.empty((2*times.size - 1,))
        new_values = np.empty_like(new_times)
        new_times[::2] = times
        new_times[1::2] = times[1:]
        new_values[::2] = values
        new_values[1::2] = values[:-1]
        return new_times, new_values
    '''
