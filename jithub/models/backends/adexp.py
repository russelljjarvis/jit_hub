from quantities import mV, ms, s, V
from neo import AnalogSignal
import numpy as np
import quantities as pq
import numpy

voltage_units = mV
import cython
from elephant.spike_train_generation import threshold_detection
import numpy as np
from numba import jit
from sciunit.models.backends import Backend
from sciunit.models import RunnableModel
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Text

# code is a very aggressive hack on this repository:
# https://github.com/ericjang/pyN, of which it now resembles very little.
@cython.boundscheck(False)
@cython.wraparound(False)
@jit(nopython=True)
def evaluate_vm(
    time_trace,
    dt,
    T,
    w,
    b,
    a,
    spike_delta,
    spike_raster,
    v_reset,
    v_rest,
    tau_m,
    tau_w,
    v_thresh,
    delta_T,
    cm,
    amp,
    start,
    stop,
):

    i = 0
    spike_raster = [0 for ix in range(0, len(time_trace))]
    vm = []
    spk_cnt = 0
    v = v_rest
    for t_ind in range(0, len(time_trace)):
        t = time_trace[t_ind]
        if i != 0:
            I_scalar = 0
        if start <= t <= stop:
            I_scalar = amp
        if spike_raster[i - 1]:
            v = v_reset
            w += b
        dv = (
            ((v_rest - v) + delta_T * np.exp((v - v_thresh) / delta_T)) / tau_m
            + (I_scalar - w) / cm
        ) * dt
        v += dv
        w += dt * (a * (v - v_rest) - w) / tau_w * dt
        if v > v_thresh:
            v = spike_delta
            spike_raster[i] = 1
            spk_cnt += 1
        else:
            spike_raster[i] = 0
        vm.append(v)
        i += 1
    return vm, spk_cnt


class JIT_ADEXPBackend():
    name = "ADEXP"

    def __init__(self, attrs={}):
        print('gets here')
        #super(JIT_ADEXPBackend,self).init_backend()
        self.vM = None
        self._attrs = attrs
        self.default_attrs = {}
        self.default_attrs["cm"] = 2.81
        self.default_attrs["v_spike"] = -40.0
        self.default_attrs["v_reset"] = -70.6
        self.default_attrs["v_rest"] = -70.6
        self.default_attrs["tau_m"] = 9.3667
        self.default_attrs["a"] = 4.0
        self.default_attrs["b"] = 0.0805
        self.default_attrs["delta_T"] = 2.0
        self.default_attrs["tau_w"] = 144.0
        self.default_attrs["v_thresh"] = -50.4
        self.default_attrs["spike_delta"] = 30
        #self.default_attrs = BAE1

        if type(attrs) is not type(None):
            self._attrs = attrs
        if not len(self._attrs):# is None:
            self._attrs = self.default_attrs
        self._vec_attrs = []

    # def as_sciunit_model(self):
    #    super().__init__(name="ADEXP")
    #    super().init_backend(attrs=self._attrs, name="ADEXP")
    #    return self

    def set_stop_time(self, stop_time=650 * pq.ms):
        """Sets the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.tstop = float(stop_time.rescale(pq.ms))

    def simulate(
        self, attrs={}, T=50, dt=0.25, integration_time=30, I_ext={}, spike_delta=50
    ) -> Tuple[Any, int]:
        """
        -- Synpopsis: simulate model
        -- outputs vm and spike count
        """
        N = 1
        w = 1
        dt = dt
        time_trace = np.arange(0, T + dt, dt)
        len_time_trace = len(time_trace)
        spike_raster = np.zeros((1, len_time_trace))
        v_rest = attrs["v_rest"]
        v_reset = attrs["v_reset"]
        tau_m = attrs["tau_m"]
        delta_T = attrs["delta_T"]
        spike_delta = attrs["spike_delta"]

        a = attrs["a"]
        b = attrs["b"]
        v_thresh = attrs["v_thresh"]
        cm = attrs["cm"]
        tau_w = attrs["tau_w"]
        amp = I_ext["pA"]
        start = I_ext["start"]
        stop = I_ext["stop"]

        vm, n_spikes = evaluate_vm(
            time_trace,
            dt,
            T,
            w,
            b,
            a,
            spike_delta,
            spike_raster,
            v_reset,
            v_rest,
            tau_m,
            tau_w,
            v_thresh,
            delta_T,
            cm,
            amp,
            start,
            stop,
        )
        return [vm, n_spikes]

    def get_spike_count(self):
        return self.n_spikes

    @property
    def attrs(self):
        return self._attrs

    @attrs.setter
    def attrs(self, attrs):
        #print(type(attrs),type(self.default_attrs))
        self.default_attrs.update(attrs)
        attrs = self.default_attrs
        self._attrs = attrs
        if hasattr(self, "model"):
            if not hasattr(self.model, "attrs"):
                self.model.attrs = {}
                self.model.attrs.update(attrs)

    def get_membrane_potential(self):
        """Must return a neo.core.AnalogSignal."""
        return self.vM

    def set_stop_time(self, stop_time=650 * pq.ms):
        """Sets the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.tstop = float(stop_time.rescale(pq.ms))
    #def load_model(self) -> None:
    #    self

    def inject_square_current(
        self,
        amplitude=100 * pq.pA,
        delay=10 * pq.ms,
        duration=500 * pq.ms,
        padding=0 * pq.ms,
    ) ->AnalogSignal:
        """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
        Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
        where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
        Description: A parameterized means of applying current injection into defined
        Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.
        """
        padding = float(padding)
        amplitude = float(amplitude.magnitude)
        duration = float(duration)
        delay = float(delay)
        tMax = delay + duration + padding

        self.set_stop_time(stop_time=tMax * pq.ms)
        tMax = float(self.tstop)

        stim = {"start": delay, "stop": duration + delay, "pA": amplitude}

        vm, n_spikes = self.simulate(attrs=self.attrs, T=tMax, dt=0.25, I_ext=stim)
        vM = AnalogSignal(vm, units=voltage_units, sampling_period=0.25 * pq.ms)

        self.vM = vM
        self.n_spikes = n_spikes

        return self.vM

    def _backend_run(self):
        results = {}
        results["vm"] = self.vM.magnitude
        results["t"] = self.vM.times
        results["run_number"] = results.get("run_number", 0) + 1
        return results

    @property
    def vector_attrs(self):
        return self._vec_attrs

    @vector_attrs.setter
    def vector_attrs(self, to_set_vec_attrs: list):
        self._vec_attrs = to_set_vec_attrs
        # stores parameters for a list of models.

    # Too hard
    from numba import vectorize

    def inject_square_current_vectorized(self, list_of_param_dicts):
        # @guvectorize([(int64[:], int64[:])], '(n)->(n)')
        # npoints = int(1e7)
        # a = np.arange(npoints,dtype=np.float32)

        def evaluate_vm_collection(arrays):
            npoints = len(arrays)
            T = 10000

            vm_returns = np.zeros( (npoints, T), dtype=np.float32 )
            #np.arange(npoints, dtype=np.float32)
            arr_cnt = 0
            dt = 1
            time_trace = np.arange(0, T + dt, dt)
            # len_time_trace = len(time_trace)
            # spike_raster = np.zeros((1, len_time_trace))

            for arr in arrays:
                w = arr[0]
                b = arr[1]
                a = arr[2]
                spike_delta = arr[3]
                v_reset = arr[4]
                v = v_rest = arr[5]
                tau_m = arr[6]
                tau_w = arr[7]
                v_thresh = arr[8]
                delta_T = arr[9]
                cm = arr[10]
                amp = arr[11]
                start = arr[12]
                stop = arr[13]
                i = 0
                # spike_raster = [0 for ix in range(0, len(time_trace))]
                vm = []
                spk_cnt = 0
                for t_ind in range(0, len(time_trace)-1):
                    t = time_trace[t_ind]
                    I_scalar = 0
                    if start <= t <= stop:
                        I_scalar = amp
                    # if spike_raster[i - 1]:
                    #    v = v_reset
                    #    w += b
                    dv = (
                        ((v_rest - v) + delta_T * np.exp((v - v_thresh) / delta_T))
                        / tau_m
                        + (I_scalar - w) / cm
                    ) * dt
                    v += dv
                    w += dt * (a * (v - v_rest) - w) / tau_w * dt
                    if v > v_thresh:
                        v = spike_delta
                        # spike_raster[i] = 1
                        spk_cnt += 1
                    else:
                        pass
                        # spike_raster[i] = 0
                    vm.append(v)
                    i += 1
                vm_returns[arr_cnt] = vm
                arr_cnt += 1
            return vm_returns
        vm_returns = evaluate_vm_collection(list_of_param_dicts)
        return vm_returns
