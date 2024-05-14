from post_analysis_module.helpers.loaded_model import model 
import typing


topic_list = ['depression',
 'relationships',
 'family',
 'romantic',
 'trauma',
 'anger-management',
 'addiction',
 'sexuality',
 'behavior']

def getPredictions(lookups: typing.List[str])->typing.List[str]:
    return [topic_list[i] for i in model.predict(lookups)]


if (__name__ == "__main__"):
    getPredictions(["Use your anger, strike him down", "i am very depressed right now"])