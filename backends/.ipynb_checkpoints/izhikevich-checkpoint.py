from sciunit.models.backends import Backend
from quantities import mV, ms, s, V
from neo import AnalogSignal
import numpy as np
import quantities as pq
import numpy
import copy

from numba import jit#, autojit
import cython


class JIT_IzhiBackend(Backend):
    def init_backend(self):
        super().init_backend()
        self.attrs = self.model.attrs
    
    
    def _backend_run(self):
        """Must return a neo.core.AnalogSignal"""
        if self.vM is not None:
            return self.vM
        else:
            everything = copy.copy(self.model.attrs)
            if hasattr(self,'Iext'):
                everything.update({'Iext':self.Iext})

            if 'current_inj' in everything.keys():
                everything.pop('current_inj',None)
            everything = copy.copy(self.model.attrs)

            self.model.attrs['celltype'] = round(self.model.attrs['celltype'])
            if self.model.attrs['celltype'] <= 3:
                everything.pop('celltype',None)
                v = get_vm_matlab_one_two_three(**everything)
            else:
                if self.model.attrs['celltype'] == 4:
                    v = get_vm_matlab_four(**everything)
                if self.model.attrs['celltype'] == 5:
                    v = get_vm_matlab_five(**everything)
                if self.model.attrs['celltype'] == 6:
                    v = get_vm_matlab_six(**everything)
                if self.model.attrs['celltype'] == 7:
                    #print('gets into multiple regimes',self.attrs['celltype'])

                    v = get_vm_matlab_seven(**everything)

            return AnalogSignal(v, units=pq.mV,
                                sampling_period=0.125*pq.ms)
        
        
    def inject_ramp_current(self, t_stop, gradient=0.000015, onset=30.0, baseline=0.0, t_start=0.0):
        times, amps = self.ramp(gradient, onset, t_stop, baseline=0.0, t_start=0.0)
 
        everything = copy.copy(self.attrs)
        
        everything.update({'ramp':amps})
        everything.update({'start':onset})
        everything.update({'stop':t_stop})

        if 'current_inj' in everything.keys():
            everything.pop('current_inj',None)

        self.attrs['celltype'] = round(self.attrs['celltype'])
        if np.bool_(self.attrs['celltype'] <= 3):
            everything.pop('celltype',None)
            v = get_vm_matlab_one_two_three(**everything)
        else:



            if np.bool_(self.attrs['celltype'] == 4):
                v = get_vm_matlab_four(**everything)
            if np.bool_(self.attrs['celltype'] == 5):
                v = get_vm_matlab_five(**everything)
            if np.bool_(self.attrs['celltype'] == 6):
                v = get_vm_matlab_six(**everything)
            if np.bool_(self.attrs['celltype'] == 7):
                v = get_vm_matlab_seven(**everything)


        self.attrs

        self.vM = AnalogSignal(v,
                            units=pq.mV,
                            sampling_period=0.125*pq.ms)

        return self.vM	
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def inject_square_current(self, amplitude=100*pq.pA, delay=10*pq.ms, duration=500*pq.ms):
        """
        Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
        Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
        where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
        Description: A parameterized means of applying current injection into defined
        Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.

        """
        
        attrs = self.model.attrs
        if attrs is None:
            attrs = self.model.default_attrs

        self.attrs = attrs
        square = True
        amplitude = float(amplitude.magnitude)
        duration = float(duration)
        delay = float(delay)
        #print(amplitude,duration,delay)
        tMax = delay + duration #+ 200.0#*pq.ms

        #self.set_stop_time(tMax*pq.ms)
        tMax = self.tstop = float(tMax)
        N = int(tMax/0.125)
        Iext = np.zeros(N)
        delay_ind = int((delay/tMax)*N)
        duration_ind = int((duration/tMax)*N)

        Iext[0:delay_ind-1] = 0.0
        Iext[delay_ind:delay_ind+duration_ind-1] = amplitude
        Iext[delay_ind+duration_ind::] = 0.0
        self.Iext = None
        self.Iext = Iext


        everything = copy.copy(self.attrs)
        everything.update({'N':len(Iext)})

        #everything.update({'Iext':Iext})
        everything.update({'start':delay_ind})
        everything.update({'stop':delay_ind+duration_ind})
        everything.update({'amp':amplitude})

        if 'current_inj' in everything.keys():
            everything.pop('current_inj',None)
        #import pdb; pdb.set_trace()
        self.attrs['celltype'] = round(self.attrs['celltype'])
        if np.bool_(self.attrs['celltype'] <= 3):
            everything.pop('celltype',None)
            v = get_vm_matlab_one_two_three(**everything)
        else:



            if np.bool_(self.attrs['celltype'] == 4):
                v = get_vm_matlab_four(**everything)
            if np.bool_(self.attrs['celltype'] == 5):
                v = get_vm_matlab_five(**everything)
            if np.bool_(self.attrs['celltype'] == 6):
                v = get_vm_matlab_six(**everything)
            if np.bool_(self.attrs['celltype'] == 7):
                v = get_vm_matlab_seven(**everything)


        self.attrs

        self.vM = AnalogSignal(v,
                            units=pq.mV,
                            sampling_period=0.125*pq.ms)
        #print(np.std(v))
        return self.vM



@jit(nopython=True)
def get_vm_matlab_four(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6,
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0,ramp=None):
    tau = dt = 0.125
    if ramp is not None:
        N = len(ramp)
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0	
        if ramp is not None:
            I = ramp[i]
        elif start <= i <= stop:
               I = amp
        # forward Euler method
        v[i+1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C
        u[i+1] = u[i]+tau*a*(b*(v[i]-vr)-u[i]); # Calculate recovery variable

        if v[i+1] > (vPeak - 0.1*u[i+1]):
            v[i] = vPeak - 0.1*u[i+1]
            v[i+1] = c + 0.04*u[i+1]; # Reset voltage
            if (u[i]+d)<670:
                u[i+1] = u[i+1]+d; # Reset recovery variable
            else:
                u[i+1] = 670;

    return v

@jit(nopython=True)
def get_vm_matlab_five(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6,
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0,ramp=None):

    tau= dt = 0.125; #dt
    if ramp is not None:
        N = len(ramp)
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0	
        if ramp is not None:
            I = ramp[i]
        elif start <= i <= stop:
            I = amp
	# forward Euler method
        v[i+1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C

        #u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]); # Calculate recovery variable
        if v[i+1] < d:
            u[i+1] = u[i] + tau*a*(0-u[i])
        else:
            u[i+1] = u[i] + tau*a*((0.125*(v[i]-d)**3)-u[i])
        if v[i+1]>=vPeak:
            v[i]=vPeak;
            v[i+1]=c;

    return v


@jit(nopython=True)
def get_vm_matlab_seven(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6,
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0,ramp=None):
    tau= dt = 0.125; #dt

    if ramp is not None:
        N = len(ramp)
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0	
        if ramp is not None:
            I = ramp[i]
        elif start <= i <= stop:
            I = amp

        # forward Euler method
        v[i+1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C


        if v[i+1] > -65:
            b=2;
        else:
            b=10;

        u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]);
        if v[i+1]>=vPeak:
            v[i]=vPeak;
            v[i+1]=c;
            u[i+1]=u[i+1]+d;  # reset u, except for FS cells


    return v

@jit(nopython=True)
def get_vm_matlab_six(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6,
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0,ramp=None):
    tau= dt = 0.125; #dt

    if ramp is not None:
        N = len(ramp)
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0	
        if ramp is not None:
            I = ramp[i]
        elif start <= i <= stop:
            I = amp
       # forward Euler method
        v[i+1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C


        u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]);
        if v[i+1] > -65:
            b=0;
        else:
            b=15;
        if v[i+1] > (vPeak + 0.1*u[i+1]):
            v[i]= vPeak + 0.1*u[i+1];
            v[i+1] = c-0.1*u[i+1]; # Reset voltage
            u[i+1]=u[i+1]+d;

    return v



@jit(nopython=True)
def get_vm_matlab_one_two_three(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6,
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,
          N=0,start=0,stop=0,amp=0,ramp=None):
    tau= dt = 0.125; #dt
    if ramp is not None:
        N = len(ramp)
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0	
        if ramp is not None:
            I = ramp[i]
        elif start <= i <= stop:
            I = amp
       # forward Euler method
        v[i+1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + I) / C
        u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]); # Calculate recovery variable
        #u[i+1]=u[i]+tau*a*(b*(v[i]-vr)-u[i]); # Calculate recovery variable

        if v[i+1]>=vPeak:
            v[i]=vPeak
            v[i+1]=c
            u[i+1]=u[i+1]+d  # reset u, except for FS cells
    return v