#metadata
import json
SOCIAL_DETERMINANT_OPTIONS = [
    "gender", 
    "race", 
    "employment status", 
    "romantic relationship status",
    "housing situation", 
    "rural vs urban",
    "neighborhood safety",
    "economic class", 
    "social acceptance", 
    "familial relations"
]

tweets_familial_relations = [
    "As the COVID-19 lockdowns lift, how will we soothe our 'Re-Entry Anxiety' #anxiety #anxietyrelief #anxietyawareness #anxietysupport #anxietyquote #covid19usa #covid2019 https://t.co/uQNWlYCT8n",
    "As someone who struggled with an abusive family as a child, this needs to be investigated and measures to ensure children safety. Please sign on here today",
    "Finally got to see Elton John!",
    "I feel very lonely right now as everyone has gone home for the Holidays - if anybody knows anything fun to do on your own please let me know."
]


import openai 
import asyncio

model_engine = "gpt-3.5-turbo"


global general_prompt
general_prompt = f"""
You are a psychologist assistance AI who will be given a tweet made by a user. From there, you will extract relavent social determiants of health. The socail determinants of health you should consider are {",".join(SOCIAL_DETERMINANT_OPTIONS[:-1])}, and {SOCIAL_DETERMINANT_OPTIONS[-1]}. Each of these determinants is for the author (i.e. the author's romatnic relationship or gender or race, not someone else they talk about). For each one you will label it as present (1) or absent (0). You will then mark if it looks like it is haivng a positive affect on the user's mental heatlh (1), a negative affect (-1), or no effect at all (0). You will also rank 1-5 how strong each present one is. 

You will return this data as a JSON object. Your JSON object will consist of the following keys: "thoughts" and "labels" and "classification" which you will generate in the order.
In "thoughts", for each social determinant given you will reason if it is present or absent and if present if it is adverse. Your labels output will be a list of relevant social determinants and can be empty if none are indicated. In classifcation, you will summarize your findings. The classification key is a dictionary mapping "therapist_help_requested" to 1 or 0. If it is 0, "problem_summary" and "solution_summary" can be empty strings, otherwise summarize the problem and possible solutions.

We sure to make this json serializable and parseable via the python json.loads function

Here is an example below:

Input:
I always feel so lonely this time of year. February really comes around to remind me that I'm not good enough right after December hammered it in. All I can do is drink my pain away, until I finally drink away all these lonely memories. The only shitty thing keeping me going is that I finally got a job, so there's a half chance if that can improve at least there's a chance everything else won't remain shitty.

Output:
{{
    "thoughts":"gender: I don't see any mention of gender. It could be that men/women tend to be sadder in February but this seems like it's absent. \
                 race: I don't see any mention of race. Again, people of certain races may tend to have months that are sadder for them but this seems like it's absent. \
                 employment status: there seems to be a positive mention of employment status as the person says at least I have a job. \
                 romantic relationship status: this seems very likely to be present as it is February and so Valentines might be around the corner. Particularly, December being rough could also indicidate this. He seems very upset so this might well be adverse. \
                 housing situation: there is no mention of housing \
                 rural vs urban: there is nothing to indicate rural vs urban \
                 neighborhood safety: there are no mentions of safety \
                 economic class: this could be causing feelings of not being enough or loneliness if have to work but it's likely not this. \
                 social acceptance: this person feels lonely so this could be part of a social status. Because of the focus on February it might be slightly less than romantic relationship status but this is still present and adverse. \
                 familial relations: this is the same as with social status. If December was rough, family situation might not be good",
    "labels":{{"romantic relationship status": {{"strength": 5, "affect": -1}},
        "social acceptance": {{"strength": 4, "affect": -1}},
        "familial relations": {{"strength": 3, "affect": -1}},
        "employment status": {{"strength": 1, "affect": 1}}
    }},
    "classification": {{
        "therapist_help_requested": 1,
        "problem_summary": "feeling lonely around holidays; particularly romantic loneliness",
        "solution_summary": "increased social connection, getting back out and dating people (self-confidence boost)"
    }}

}}
"""





#TODO - improve for now just put keys in 
def enforce_format(d):
    for _key in ["classification", "labels"]:
        assert(_key in d)
    for v in d["labels"].values():
        #for each key, assert the proper values are in it
        assert("strength" in v)
        assert("affect" in v)

import typing
from typing import Awaitable, TypedDict

class SdohClassification:
    """
    therapist_help_requested \in [0,1] representing True/False
    """
    solution_summary:str 
    problem_summary:str 
    therapist_help_requested: int

class SdohStrength(TypedDict):
    strength:int
    affect:int


class SdohResponse(TypedDict):
    labels: typing.Dict[str, SdohStrength]
    thoughts:str
    affect:SdohClassification

#TODO - make async
async def get_responses(submisison_body:str, api_key:str, cnt:int = 4)->Awaitable[SdohResponse]:
    """
    @params
        submission_body: the submission to gpt
        api_key: GPT API key to use
        cnt: the number of times to retry
    """
    openai.api_key = api_key
    global general_prompt
    completion = await openai.ChatCompletion.acreate(
                model = model_engine,
                messages = [{"role": "system", "content": general_prompt}] +[{"role": "user", "content": sp} for sp in [submisison_body]]
                            
        )
    txt = completion.choices[0].message.content.strip()
    try:
        json_res = json.loads(txt)
        enforce_format(json_res)
        return json_res 
    except:
        if (cnt !=0):
            return await get_responses(submisison_body, cnt -1)
        else:
            return dict()
        
async def amain():
    print(await get_responses("I feel depressed because I have too many expectations. My parents are too strict","LOAD API KEY HERE"))
if (__name__ == "__main__"):
    print("sdoh main")
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(amain())