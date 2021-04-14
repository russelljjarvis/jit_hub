from neo import AnalogSignal
import numpy as np
import quantities as pq
import numpy
import cython
from elephant.spike_train_generation import threshold_detection
from numba import jit
from sciunit.models.backends import Backend
from sciunit.models import RunnableModel
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Text
from numba import float64, float32, guvectorize
from numba import guvectorize, jit, float64, void

# code once originated with this repository:
# https://github.com/ericjang/pyN, of which it now resembles very little.
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


##
# this must be a sqaure.
##
# @guvectorize([(float64[:,:], float64[:,:], float64[:,:])], '(m,l),(l,n)->(m,n)', target='cpu')

# @guvectorize([(float32[:,:], float32[:,:])], '(m,l),(l,n)->(m,n)')#, target='cuda')

# @guvectorize([(float64[:,:], float64[:,:], float64[:,:])], '(m,l),(l,n)->(m,n)', target='cpu')
# @jit#([float64vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv  c ghf n[:,:], float64[:,:]])
#@jit
#@guvectorize(['(float32[:,:], float32[:,:])'],'(n,m)->(n,l)', target='cpu')
#
#     ufunc = guvectorize(['(int32[:,:], int32[:,:], int32[:,:])'],
#                         "(x,y),(x,y),(x,y)")(guadd)#

#  https://towardsdatascience.com/heres-how-to-use-cupy-to-make-numpy-700x-faster-4b920dda1f56
'''
import cupy as cp

x_gpu = cp.ones((1000,1000,1000))
cp.cuda.Stream.null.synchronize()
'''
#@guvectorize(['(int32[:,:], int32[:,:])'])
def evaluate_vm_collection(arr, vm):
    n = arr.shape[0]
    print(np.shape(arr),'in shape')
    for i in range(n):
        cm = arr[i, 0]
        factored_out = arr[i, 1]
        v_reset = arr[i, 2]
        v = v_rest = arr[i, 3]
        tau_m = arr[i, 4]
        a = arr[i, 5]
        b = arr[i, 6]
        delta_T = arr[i, 7]
        tau_w = arr[i, 8]
        v_thresh = arr[i, 9]
        spike_delta = arr[i, 10]
        dt = arr[i, 11]
        start = arr[i, 12]
        stop = arr[i, 13]
        amp = arr[i, 14]
        padding = arr[i, 15]
        tMax = start + stop + padding
        time_trace = np.arange(0, int(tMax + dt), dt)
        spike_raster = np.zeros(len(time_trace))
        len_time_trace = len(time_trace)
        sim_len = int(len(time_trace)) - 1
        arr_cnt = 0
        w = 1
        spk_cnt = 0
        for t_ind in range(0, len_time_trace - 1):
            t = time_trace[t_ind]
            I_scalar = 0
            if start <= t <= stop:
                I_scalar = amp

            if spike_raster[t_ind]:
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
                spike_raster[t_ind] = 1

                spk_cnt += 1
            else:
                spike_raster[t_ind] = 0
            vm[i, t_ind] = v
    print(np.shape(vm),'out shape')

    #return vm


class JIT_ADEXPBackend:
    name = "ADEXP"

    def __init__(self, attrs={}):
        self.vM = None
        self._attrs = attrs
        self.default_attrs = {}
        self.default_attrs["cm"] = 2.81
        # self.default_attrs["v_spike"] = -40.0
        self.default_attrs["v_reset"] = -70.6
        self.default_attrs["v_rest"] = -70.6
        self.default_attrs["tau_m"] = 9.3667
        self.default_attrs["a"] = 4.0
        self.default_attrs["b"] = 0.0805
        self.default_attrs["delta_T"] = 2.0
        self.default_attrs["tau_w"] = 144.0
        self.default_attrs["v_thresh"] = -50.4
        self.default_attrs["spike_delta"] = 30
        self.default_attrs["dt"] = 0.25

        if type(attrs) is not type(None):
            self._attrs = attrs
        if not len(self._attrs):  # is None:
            self._attrs = self.default_attrs
        self._vec_attrs = []

    def set_stop_time(self, stop_time=650 * pq.ms):
        """Sets the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.tstop = float(stop_time.rescale(pq.ms))

    def simulate(
        self, attrs={}, T=50, dt=0.25, I_ext={}, spike_delta=50
    ):  # -> Tuple[Any, int]:
        """
        -- Synpopsis: simulate model
        -- outputs vm and spike count
        """
        N = 1
        w = 1
        time_trace = np.arange(0, T, dt)
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
        self.default_attrs.update(attrs)
        attrs = self.default_attrs
        self._attrs = attrs
        if hasattr(self, "model"):
            if not hasattr(self.model, "attrs"):
                self.model.attrs = {}
                self.model.attrs.update(attrs)

    def set_attrs(self, attrs):
        self.attrs = attrs

    def get_membrane_potential(self):
        """Must return a neo.core.AnalogSignal."""
        return self.vM

    def set_stop_time(self, stop_time=650 * pq.ms):
        """Sets the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.tstop = float(stop_time.rescale(pq.ms))

    def inject_square_current(
        self,
        amplitude=100 * pq.pA,
        delay=10 * pq.ms,
        duration=500 * pq.ms,
        padding=0 * pq.ms,
        dt=0.25,
    ):  # -> AnalogSignal:
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
        if "dt" in self.attrs:
            self.attrs["dt"]
        else:
            dt = 0.1
        vm, n_spikes = self.simulate(attrs=self.attrs, T=tMax, dt=dt, I_ext=stim)
        vM = AnalogSignal(vm, units=pq.mV, sampling_period=dt * pq.ms)

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

    @jit
    def make_gene_array(self, gene_models):
        """"""
        param_vec = np.array(
            [
                np.array([v for v in list(m.attrs.values())], dtype=np.float32)
                for m in gene_models
            ]
        )
        # self._backend.vector_attrs = param_vec
        return param_vec

    def eval_models_as_gene_array(self, models):
        param_vec = self.make_gene_array(models)
        results = self.inject_square_current_vectorized(param_vec)

    @jit
    def inject_square_current_vectorized(self, gene_param_arr):

        dt = gene_param_arr[0, 11]
        start = gene_param_arr[0, 12]
        stop = gene_param_arr[0, 13]
        padding = gene_param_arr[0, 15]
        tMax = start + stop + padding
        time_trace = np.arange(0, int(tMax + dt), dt)
        volts = [-65.0 for i in range(0, len(time_trace))]
        vm = np.array([volts for i in range(0, np.shape(gene_param_arr)[0])])
        vm_returned = evaluate_vm_collection(gene_param_arr, vm)
        return vm_returned
