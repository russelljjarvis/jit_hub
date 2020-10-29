
from quantities import mV, ms, s, V
from neo import AnalogSignal
import neuronunit.capabilities as cap
import numpy as np
import quantities as pq
import numpy
voltage_units = mV

from elephant.spike_train_generation import threshold_detection

from numba import jit, autojit
import cython
@jit(nopython=True)
def get_vm_matlab_four(C=89.7960714285714,
         a=0.01, b=15, c=-60, d=10, k=1.6, 
         vPeak=(86.364525297619-65.2261863636364),
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0):
    tau = dt = 0.125
    #I = 0
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0

        if start <= i <= stop:
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
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0):
    
    tau= dt = 0.125; #dt
    #I = 0
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0

        if start <= i <= stop:
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
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0):
    tau= dt = 0.125; #dt

    #I = 0
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0

        if start <= i <= stop:
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
          vr=-65.2261863636364, vt=-50,celltype=1, N=0,start=0,stop=0,amp=0):
    tau= dt = 0.125; #dt

    #I = 0
    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0

        if start <= i <= stop:
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
          N=0,start=0,stop=0,amp=0):
    tau= dt = 0.125; #dt

    v = np.zeros(N)
    u = np.zeros(N)
    v[0] = vr
    for i in range(N-1):
        I = 0

        if start <= i <= stop:
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
        
    

class IZHIModel():

    name = 'IZHI'

    def __init__(self, attrs=None,DTC=None,
                     debug = False):
        self.vM = None
        self.attrs = attrs
        self.temp_attrs = None
        self.default_attrs = {'C':89.7960714285714, 
            'a':0.01, 'b':15, 'c':-60, 'd':10, 'k':1.6, 
            'vPeak':(86.364525297619-65.2261863636364), 
            'vr':-65.2261863636364, 'vt':-50, 'celltype':3}

        if type(attrs) is not type(None):
            self.attrs = attrs
        if self.attrs is None:
            self.attrs = self.default_attrs


    def get_spike_count(self):
        thresh = threshold_detection(self.vM,0*qt.mV)
        return len(thresh)


    def set_stop_time(self, stop_time = 650*pq.ms):
        """Sets the simulation duration
        stopTimeMs: duration in milliseconds
        """
        self.tstop = float(stop_time.rescale(pq.ms))


    def get_membrane_potential(self):
        """Must return a neo.core.AnalogSignal.
        """
        if type(self.vM) is not type(None):
            return self.vM


        if type(self.vM) is type(None):

            everything = copy.copy(self.attrs)
            if hasattr(self,'Iext'):
                everything.update({'Iext':self.Iext})
            
            if 'current_inj' in everything.keys():
                everything.pop('current_inj',None)
            everything = copy.copy(self.attrs)

            self.attrs['celltype'] = round(self.attrs['celltype'])
            if self.attrs['celltype'] <= 3:   
                everything.pop('celltype',None)         
                v = get_vm_matlab_one_two_three(**everything)
            else:
                if self.attrs['celltype'] == 4:
                    v = get_vm_matlab_four(**everything)
                if self.attrs['celltype'] == 5:
                    v = get_vm_matlab_five(**everything)
                if self.attrs['celltype'] == 6:
                    v = get_vm_matlab_six(**everything)
                if self.attrs['celltype'] == 7:
                    #print('gets into multiple regimes',self.attrs['celltype'])

                    v = get_vm_matlab_seven(**everything)

            self.vM = AnalogSignal(v,
                                units=pq.mV,
                                sampling_period=0.125*pq.ms)

            
        return self.vM

    def set_attrs(self, attrs):
        self.attrs = attrs
        self.model.attrs.update(attrs)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def inject_square_current(self, current):
        """Inputs: current : a dictionary with exactly three items, whose keys are: 'amplitude', 'delay', 'duration'
        Example: current = {'amplitude':float*pq.pA, 'delay':float*pq.ms, 'duration':float*pq.ms}}
        where \'pq\' is a physical unit representation, implemented by casting float values to the quanitities \'type\'.
        Description: A parameterized means of applying current injection into defined
        Currently only single section neuronal models are supported, the neurite section is understood to be simply the soma.

        """
        
        attrs = copy.copy(self.model.attrs)
        if attrs is None:
            attrs = self.default_attrs

        self.attrs = attrs
        if 'injected_square_current' in current.keys():
            c = current['injected_square_current']
        else:
            c = current
        amplitude = float(c['amplitude'])
        duration = float(c['duration'])
        delay = float(c['delay'])
        tMax = delay + duration + 200.0

        self.set_stop_time(tMax*pq.ms)
        tMax = self.tstop

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
        

        self.model.attrs.update(attrs)

        self.vM = AnalogSignal(v,
                            units=pq.mV,
                            sampling_period=0.125*pq.ms)

    def _backend_run(self):
        results = {}
        if len(self.attrs) > 1:
            v = get_vm(**self.attrs)
        else:
            v = get_vm(self.attrs)

        self.vM = AnalogSignal(v,
                               units = voltage_units,
                               sampling_period = 0.01*pq.ms)
        results['vm'] = self.vM.magnitude
        results['t'] = self.vM.times
        results['run_number'] = results.get('run_number',0) + 1
        return results
  
