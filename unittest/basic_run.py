import jithub
from jithub.models import model_classes
import quantities as pq
cellmodels = ["IZHI","MAT","ADEXP"]
import unittest
for cellmodel in cellmodels:
    if cellmodel == "IZHI":
        model = model_classes.IzhiModel()
    if cellmodel == "MAT":
        model = model_classes.MATModel()
    if cellmodel == "ADEXP":
        model = model_classes.ADEXPModel()
    print(cellmodel)
    ALLEN_DELAY = 1000.0 * pq.ms
    ALLEN_DURATION = 2000.0 * pq.ms
    uc = {
        "amplitude": 25*pq.pA,
        "duration": ALLEN_DURATION,
        "delay": ALLEN_DELAY,
    }
    model.inject_square_current(**uc)
    vm = model.get_membrane_potential()
    try:
        assert vm.times[-1] == ALLEN_DELAY + ALLEN_DURATION
    except:
        print(vm.times[-1],ALLEN_DELAY + ALLEN_DURATION)
