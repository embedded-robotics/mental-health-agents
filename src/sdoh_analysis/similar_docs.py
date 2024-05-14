import typing

from datasets import load_dataset
from post_analysis_module.helpers.html_cleaner import strip_tags
import os

import numpy as np 
from tqdm import tqdm
from post_analysis_module.helpers.loaded_model import model

from joblib import Memory
memory = Memory("datasets")
@memory.cache
def get_data(name):
    return load_dataset(name)

docs1 = get_data("nbertagnolli/counsel-chat")["train"]

#these also have the answers
# docs2 = load_dataset("loaiabdalslam/counselchat")["train"]


#initialize our db
db = list()
search_str_to_answers = dict()


numReachedLong = 0
numPastLong = 0
print("encoding database")
for documents in [docs1]:
    for title, text, answer, upvotes in tqdm(zip(documents["questionTitle"], documents["questionText"], documents["answerText"],documents["upvotes"])):
        if (not title): title = "" 
        if (not text): text = ""
        if (not answer): continue
        search_string = title + "\n\n\n" + text 
        search_string = search_string.strip()
        if (not search_string): continue 
        numReachedLong += 1
        if (len(search_string) > 5*300): continue 
        
        answer_txt = answer
        # print("answer is", answer)
        # print(len(answer_txt))
        if (len(answer_txt) > 5*400): continue
        numPastLong += 1
        db.append(search_string)

        
        # print("answer txt is ", answer_txt)
        search_str_to_answers[search_string] = search_str_to_answers.get(search_string, list())
        search_str_to_answers[search_string].append((upvotes, answer_txt))
print("got",numPastLong,"of",numReachedLong, "others out bc of size")

from tqdm import tqdm
for k, v in tqdm(search_str_to_answers.items()):
    answer_txt_to_highest_num_upvotes = dict()
    for upvotres, txt in v:
        answer_txt_to_highest_num_upvotes[txt] = max(answer_txt_to_highest_num_upvotes.get(txt,0), upvotes)
    
    new_list = [(v2,k2) for k2, v2 in answer_txt_to_highest_num_upvotes.items()]
    new_list.sort()
    new_list.reverse()
    search_str_to_answers[k] = new_list

# db = db[:100]
db = list(set(db))

print("encodnig databsae - this can take a while - TODO - cache this")
database = model.model_body.encode((db))
print("db encoded")

def get_similar_docs(query_text_list: typing.List[str], k:int =5)-> typing.List[typing.List[typing.Tuple[float, str, typing.List[typing.Tuple[int, str]]]]]:
    """
    params
        query_text_list: a list of all the texts you want to get matched documents for 
        k: the number of matches to return for each doc
    
    returns
        a list first indexable by query_text_list (first entry corresponds to first query in query_text_list)
            the second list corresponds to which of the k is being returned
                the next is a tuple who's entries are [similarity score, questionText, answerList]
                    answerList is a list of (numUpvotes, answerText)
    """

    q_encoded = model.model_body.encode(query_text_list)
    similarity = np.matmul(database, q_encoded.T)

    toRet = list()
    for sim in similarity.T:

        weakest_to_highest = np.argsort(sim,axis=None)
        total_num = similarity.shape[0]
        answer_bundles = list()

        for i in range(1,k+1):
            this_idx = total_num - i 

            similar_doc = db[weakest_to_highest[this_idx]]
            answers = search_str_to_answers[similar_doc]
            answer_bundles.append((sim[weakest_to_highest[this_idx]].item(),similar_doc, answers ))

        toRet.append(answer_bundles)
    return toRet

# print(get_similar_docs(["I wish I was dead", "I am always hungry"], 2))