import jithub
from jithub.models import model_classes

cellmodels = ["IZHI","MAT","ADEXP"]
for cellmodel in cellmodels:
    if cellmodel == "IZHI":
        model = model_classes.IzhiModel()
    if cellmodel == "MAT":
        model = model_classes.MATModel()
    if cellmodel == "ADEXP":
        model = model_classes.ADEXPModel()
