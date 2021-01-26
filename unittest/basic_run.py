import jithub
from jithub.models import model_classes
import quantites as pq
cellmodels = ["IZHI","MAT","ADEXP"]
for cellmodel in cellmodels:
    if cellmodel == "IZHI":
        model = model_classes.IzhiModel()
    if cellmodel == "MAT":
        model = model_classes.MATModel()
    if cellmodel == "ADEXP":
        model = model_classes.ADEXPModel()
ALLEN_DELAY = 1000.0 * pq.ms
ALLEN_DURATION = 2000.0 * pq.ms
uc = {
    "amplitude": 25*pq.pA,
    "duration": ALLEN_DURATION,
    "delay": ALLEN_DELAY,
}
model.inject_square_current(uc)
vm = model.get_membrane_potential()
assert vm.times[-1] == ALLEN_DELAY + ALLEN_DURATION
