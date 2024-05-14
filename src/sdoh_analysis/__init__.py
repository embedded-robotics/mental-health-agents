from post_analysis_module.sdoh import get_responses as gpt_generate_sdoh, SdohStrength, SdohResponse
import asyncio

from post_analysis_module.similar_docs import get_similar_docs  

from post_analysis_module.topic_matching import getPredictions as get_topic_predictions
import typing
import json


#define our get prompt 
def get_prompt_from_topic(topic):
    assert(topic in ['depression','relationships', 'family','romantic','trauma','anger-management','addiction','sexuality','behavior'])

    #A person is suffering from ill mental health because of {determinant} and is at a high risk of developing suicidal thoughts

    if (topic == "depression"):
        return "A person is suffering from ill mental health, likely due to depression"
    elif topic== "relationships":
        return "A person is struggling with their relationships right now"
    elif topic == "family":
        return "A person is suffering ill mental health due to familial causes"
    elif topic == "romantic":
        return  "A person is suffering ill mental health due to happenstance in their romantic life"
    elif topic == "trauma":
        return "A person is suffering from ill mental health which has the cause of an underlying traumatic experience"
    elif topic== "anger-management":
        return "A person is struggling with anger management"
    elif topic== "addiction":
        return "A person is struggling with addiction, its affects, or the affects of others in their life being addicted"
    elif topic== "sexuality":
        return "A person is struggling mentally due to issues surrounding sexuality"
    elif topic== "behavior":
        return "A person is struggling mentally due to ill-defined behavior of others"
    else:
        raise Exception("switch statement should have gotten it")

    
def get_prompt_from_social_determinants(sdoh:typing.Dict[str,SdohStrength])->str:
    if (not len(sdoh)): return ""
    sdoh_str = ""
    sdoh_list = list(sdoh.keys())
    if (len(sdoh_list) ==1):
        sdoh_str = sdoh_list[0]
    elif (len(sdoh_list) ==2):
        sdoh_str = sdoh_list[0] + " and " + sdoh_list[1]
    else:
        sdoh_str = ", ".join(sdoh_list[:-1]) + ", and " + sdoh_list[-1]
    return f"This person has indicated through this writing and others that the following social determinants of health are of importance to them: {sdoh_str}."


class ProcessTextReturn(typing.TypedDict):
    prompt:str
    sdoh: typing.List[typing.Union[SdohResponse,None]]
    matchedDocs: typing.List[typing.Tuple[float, str, typing.List[typing.Tuple[int, str]]]]
    topic:str

    


def process_text(text_process:typing.List[str],openai_api_key:str="", text_history:typing.List[typing.List[str]]=[], k = 3)->typing.List[ProcessTextReturn]:
    """
    @params
        text_process: texts to generate the prompt for, each indexis a different example
        openai_api_key: string to access openai to get sdoh
        text_history: a list to run sdoh on matchin w/ text_proces
        k: number of examples to return for each of the texts passed in 
    @returns 
        List for each text where 
            prompt: the prompt to add at beginnong of LLM
            sdoh: social determinants response. Ignore the thoughts keyword in most cases.
                classification: summary of classification
                labels: map each social determinant to its strength and affect
            matchedDocs: a list of all documents matched to this specific text
                [score, question text, [num upvotes, answer text]]
            topic: the topic detected by the model

    """

    if (not openai_api_key):
        topics = get_topic_predictions(text_process)
        prompts = [get_prompt_from_topic(topic) for topic in topics]
        matchedDocuments = get_similar_docs(text_process, k)
        sdohs = [[None]]*len(text_process)
        toRet = list()
        for topic, prompt, matchedDocs, sdoh in zip(topics, prompts, matchedDocuments, sdohs):
            toRet.append({
                "prompt": prompt, 
                "sdoh": sdoh,
                "matchedDocs": matchedDocs,
                "topic": topic
            })
        return toRet

    if (len(text_history) == 0): text_history = [[t] for t in text_process]
    assert(len(text_history) == len(text_process))

    print("generating sdoh prompts")
    text_history_sdoh_promises= [ [gpt_generate_sdoh(history, openai_api_key) for history in histories] for histories in text_history]
    print("getting topics")
    topics = get_topic_predictions(text_process)
    print("getting similar docs")
    matchedDocuments = get_similar_docs(text_process, k)
    sdohs = list()
    sdoh_prompts = list()

    #for each of the history get all sdohs there and log them
    print("resolving sdoh promises")
    for text_instance_sdohs in text_history_sdoh_promises:
        sdohs_here = list()
        sdoh_total_map = dict()
        for sdoh_promises in text_instance_sdohs:
            sdoh = sdoh_promises
            sdohs_here.append(sdoh)
            sdoh_total_map.update(sdoh)
        sdohs.append(sdohs_here)
        sdoh_prompts.append(get_prompt_from_social_determinants(sdoh_total_map["labels"]))

    print("creating prompts")
    print("sdoh prompts r ", sdoh_prompts)
    print("topics are", topics)

    prompts = [get_prompt_from_topic(topic)+sdoh_prompt_part for topic, sdoh_prompt_part in zip(topics, sdoh_prompts)]
    toRet = list()
    for topic, prompt, matchedDocs, sdoh in zip(topics, prompts, matchedDocuments, sdohs):
        toRet.append({
                "prompt": prompt, 
                "sdoh": sdoh,
                "matchedDocs": matchedDocs,
                "topic": topic
            })
    return toRet

async def amain():
    res = await process_text(["I am depressed because my father beats me", "I can't eat because I'm too fat"], "ADD GPT KEY HERE - amain of __init__.py", [
        ["The courts gave custody to my father becaue he's my biological father - I've never met him and wish I could live with my Aunt instead. I'm scared.", "I am depressed because my father beats me"],
        ["My girlfriend just dumped me - turned out she was seeing someone else. I know I'm not the most attractive girl, but we'd been together for 5 years. I just want to end things.", "I can't eat because I'm too fat"],
        ],
        add_social_determinants=True,
        k=2
    )
    print(res)
    # res = await process_text(["I am Groot", "You are Groot"])
    # print(res)
    with open("tmp.json", 'w') as f:
        json.dump(res, f)

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(amain())