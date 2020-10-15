

from quantities import mV, ms, s, V
from neo import AnalogSignal
import numpy as np
import quantities as qt

import quantities as pq
import numpy
voltage_units = mV
import cython

from elephant.spike_train_generation import threshold_detection


import numpy as np
from numba import jit

import numpy as np
from numba import jit

# code is a very aggressive
# hack on this repository:
# https://github.com/ericjang/pyN, of which it now resembles very little.
@jit
def update_currents(amp, i, t, dt,start,stop):   
  scalar = 0
  if start <= t <= stop:
    scalar = amp
  return scalar

@cython.boundscheck(False)
@cython.wraparound(False)
@jit(nopython=True)
def update_state(T, dt,v,w,
                        v_reset,b,a,spike_raster,
                        spike_delta,v_rest,tau_m,
                        tau_w,v_thresh,delta_T,cm,time_trace,amp,start,stop):
  i = 0
  spike_raster = [0 for ix in range(0,len(time_trace))]
  vm = []
  
  for t_ind in range(0,len(time_trace)):
    t = time_trace[t_ind]
    if i!=0:
      I_scalar = 0
      if start <= t <= stop:
        I_scalar = amp

      if spike_raster[i-1]:
        v = v_reset
        w += b
      dv  = (((v_rest-v) + \
            delta_T*np.exp((v - v_thresh)/delta_T))/tau_m + \
            (I_scalar - w)/cm) *dt
      v += dv
      w += dt * (a*(v - v_rest) - w)/tau_w * dt
      if v>v_thresh:
        v = spike_delta
        spike_raster[i] = 1
      else:
        spike_raster[i] = 0
      vm.append(v)
    i+=1
  return len(spike_raster),vm
@jit#(nopython=True)
def evaluate_vm(time_trace,dt,T,v,w,b,a,spike_delta,spike_raster,v_reset,v_rest,tau_m,tau_w,v_thresh,delta_T,cm,amp,start,stop):
  n_spikes,vm = update_state(T=T, 
                                    dt=dt,
                                    v=v,
                                    w=w,
                                    spike_raster=spike_raster,
                                    v_reset=v_reset,
                                    b=b,a=a,
                                    spike_delta=spike_delta,
                                    v_rest=v_rest,tau_m=tau_m,tau_w=tau_w,
                                    v_thresh=v_thresh,
                                    delta_T =delta_T,cm=cm,time_trace=time_trace,
                                    amp = amp,start = start,stop = stop)

  
  return vm,n_spikes



class ADEXP():
  name = 'ADEXP'
  def __init__(self, attrs={}):
    self.attrs = attrs
    self.vM = None
    self.temp_attrs = None
    
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

  def simulate(self, attrs={}, T=50,dt=0.25,integration_time=30, I_ext={},spike_delta=50):
    spike_delta = spike_delta
    N = 1
    w = 1

    dt         = dt
    time_trace = np.arange(0,T+dt,dt)#time array
    #I_ext
    len_time_trace = len(time_trace)
    integration_time = 30.0
    spike_raster = np.zeros((1, len_time_trace))
    integrate_window = np.int(np.ceil(integration_time/dt))

    if integrate_window > len_time_trace:
      integrate_window = len_time_trace #if we are running a short simulation then integrate window will overflow available time slots!
    attrs = self.default_attrs
    attrs.update(self.model.attrs)
    v_rest =  attrs['v_rest']
    v = v_rest 
    #v = v_rest #voltage trace
    v_reset = attrs['v_reset']
    tau_m = attrs['tau_m']
    delta_T = attrs['delta_T']
    spike_delta = attrs['spike_delta']

    a = attrs['a']
    b = attrs['b']
    v_thresh = attrs['v_thresh']
    cm = attrs['cm']
    tau_w = attrs['tau_w']

    
    amp = I_ext['pA']
    start = I_ext['start']
    stop = I_ext['stop']

    vm,n_spikes = evaluate_vm(time_trace,dt,T,v,w,b,a,
                      spike_delta,spike_raster,v_reset,v_rest,
                      tau_m,tau_w,v_thresh,delta_T,cm,amp,start,stop)
    #print(vm)
    return vm,n_spikes


  def get_spike_count(self):
    return self.n_spikes

  def set_attrs(self,attrs):
      self.default_attrs.update(attrs)
      attrs = self.default_attrs
      self.attrs = attrs
      if not hasattr(self.model,'attrs'):
        self.model.attrs = {}
      self.model.attrs.update(attrs)


  def get_membrane_potential(self):
    """Must return a neo.core.AnalogSignal.
    """
    return self.vM
  def set_stop_time(self, stop_time = 650*pq.ms):
      """Sets the simulation duration
      stopTimeMs: duration in milliseconds
      """
      self.tstop = float(stop_time.rescale(pq.ms))
  def inject_square_current(self,current):#, section = None, debug=False):
    """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
    Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
    where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
    Description: A parameterized means of applying current injection into defined
    Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.
    """

    temp_attrs =  self.attrs

    if 'injected_square_current' in current.keys():
        c = current['injected_square_current']
    else:
        c = current
    amplitude = float(c['amplitude'])
    duration = float(c['duration'])
    delay = float(c['delay'])
    tMax = delay + duration + 200.0

    self.set_stop_time(stop_time = tMax*pq.ms)
    tMax = float(self.tstop)

    stim = {'start':delay,'stop':duration+delay,'pA':amplitude}

    vm,n_spikes = self.simulate(
        attrs=temp_attrs,\
        T=tMax,\
        dt=0.25,\
        I_ext=stim)
    vM = AnalogSignal(vm,
                          units = voltage_units,
                          sampling_period = 0.25*pq.ms)

    self.vM = vM
    self.n_spikes = n_spikes
    return vM
