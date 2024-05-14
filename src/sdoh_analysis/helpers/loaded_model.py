import os 
from setfit import SetFitModel 

setfit_model_path = os.path.join(os.path.dirname(__file__), "..","models", "setfit-mental-health-topic-prediction","setfit-small-research-topic-prediction","content", "setfit-small-research-topic-prediction")
print("path is ", setfit_model_path)
print(os.path.isdir(setfit_model_path))
model = SetFitModel._from_pretrained(setfit_model_path)