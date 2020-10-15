import matplotlib.pyplot as plt
import collections
import quantities as pq
import izhi
AMPL = 10*pq.pA 
DELAY = 200*pq.ms
DURATION = 500 *pq.ms

# https://www.izhikevich.org/publications/spikes.htm
type2007 = collections.OrderedDict([
  #              C    k     vr  vt vpeak   a      b   c    d  celltype
  ('RS',        (100, 0.7,  -60, -40, 35, 0.01,   -2, -50,  100,  1)),
  ('IB',        (150, 1.2,  -75, -45, 50, 0.1,   5, -56,  130,   2)),
  ('TC',        (200, 1.6,  -60, -50, 35, 0.1,  15, -60,   10,   6)),
  ('TC_burst',  (200, 1.6,  -60, -50, 35, 0.1,  15, -60,   10,   6)),
  ('LTS',       (100, 1.0,  -56, -42, 40, 0.01,   8, -53,   20,   4)),
  ('CH',        (50,  1.5,  -60, -40, 25, 0.01,   1, -40,  150,   3))])

# http://www.physics.usyd.edu.au/teach_res/mp/mscripts/
# ns_izh002.m
# Fast spiking cannot be reproduced as it requires modifications to the standard Izhi equation,
# which are expressed in this mod file.
# https://github.com/OpenSourceBrain/IzhikevichModel/blob/master/NEURON/izhi2007b.mod

trans_dict = collections.OrderedDict([(k,[]) for k in ['C','k','vr','vt','vPeak','a','b','c','d']])

for i,k in enumerate(trans_dict.keys()):
    for v in type2007.values():
        trans_dict[k].append(v[i])


reduced_cells = collections.OrderedDict([(k,[]) for k in ['RS','IB','LTS','TC','TC_burst']])

params = {}
params['injected_square_current'] = {}
params['injected_square_current']['amplitude'] = 150*pq.pA
params['injected_square_current']['delay'] = DELAY
params['injected_square_current']['duration'] = DURATION

for index,key in enumerate(reduced_cells.keys()):
    reduced_cells[key] = {}
    for k,v in trans_dict.items():
        reduced_cells[key][k] = v[index]
    
    model = izhi.IZHIModel()
    model.set_attrs(reduced_cells[key])
    model.inject_square_current(params)
    vm = model.get_membrane_potential()
    plt.plot(vm.times,vm.magnitude)
plt.show()
