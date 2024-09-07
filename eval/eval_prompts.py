system_cot_prompt = "You are a mental health expert who specializes in providing counselling advice to mentally ill patients such that your advice aligns with counselling/psychotherapy factors"
user_cot_prompt = '''You are given a user input shared by a mentally ill person on the social media. Your job is to provide comprehensive counselling advice to this person such that it improves the overall mental health of the person by alleviating the feelings of depression, sadness, or hopelessness from the person's mind.
While crafting your counselling advice, you must adopt the practices aligning with the following psychotherapy factors (brief description is provided for each factor):

1. Creativity -> Counselling advice must be unique to the user’s situation and different than an advice given for any generic concern
2. Directedness -> Counselling advice must be direct in bringing up and discussing solutions to the problems
3. Perspective Change -> Counselling advice must focuse on being positive, thinking about others, or looking towards the future
4. Affirmations -> Counselling advice must affirm the patients experience
5. Sensitivity -> Counselling advice must be warm, accepting, and understanding to the patient
6. Empathy -> Counselling advice must depict empathy to the user
7. Persuasion -> Counselling advice should be persuasive enough for the patient to take the suggested action

Your final output must contain only the counselling advice which you crafted in context of user input and aforementioned psychotherapy factors.

User Input: {user_input}

Counselling Advice:
'''

system_rank_prompt = "You are a mental health expert who specializes in evaluating counselling advice provided to mentally ill patients based on psychotherapy factors"
user_rank_prompt = '''You are given a user input shared by a mentally ill person on the social media. Moreover, you are given two counselling responses provided to this person by mental health professionals namely "LLAMA3_Response" and "GPT4_Response"
Your job is to decide which response is most promising to alleviate feelings of depression, sadness, or hopelessness from this person's mind and improve the overall mental health state of this person.
Analyze both responses in detail and then rank the responses in context of each of the 7 psychotherapy factors given below (brief description is provided for each factor):

1. Creativity -> How unique this response is to the user’s situation and how different is this than a response to a generic concern
2. Directedness -> How direct is this response in bringing up and discussing solutions to the problems
3. Perspective Change -> How much this response focuses on being positive, thinking about others, or looking towards the future
4. Affirmations -> How much they affirm the patients experience
5. Sensitivity -> How warm, accepting, and understanding this response is to the patient
6. Empathy -> How much empathy this response shows to the user
7. Persuasion -> How persuasive this response is for the patient to take the suggested action

Your final output must create the key-value pairs in json format for each Psychotherapy factor, and it should be json serializable as following:

<Psychotherapy_Factor>: <List of Responses ranked in context of relevant Psychotherapy Factor>

An example response must look like the following:
{
    "Creativity": [LLAMA3_Response, GPT4_Response],
    "Directedness": [GPT4_Response, LLAMA3_Response],
    "Perspective Change": [LLAMA3_Response, GPT4_Response],
    "Affirmations": [LLAMA3_Response, GPT4_Response],
    "Sensitivity": [GPT4_Response, LLAMA3_Response],
    "Empathy": [LLAMA3_Response, GPT4_Response],
    "Persuasion": [GPT4_Response, LLAMA3_Response]
}

User Input: {user_input}
LLAMA3_Response: {llama3_response}
GPT4_Response: {gpt4_response}
Output:
'''
