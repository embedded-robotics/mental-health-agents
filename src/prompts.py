standard_prompt = '''
Provide mental health counselling to the person sharing the following thoughts on social media: {input}
'''

cot_prompt = '''
A person is suffering from ill mental health and is at a high risk of developing suicidal thoughts. The person expresses the internal feelings on social media as follows: {input}
Your job is to provide sincere counselling to this person to forestall the generation of suicidal thoughts. You firstly need to make a comprehensive plan and then write a counselling advice. Your output should be of the following format:

Plan:
Your plan here.

Counsel:
Your counsel here.
'''

cot_prompt_dynamic = '''
{det_prompt}. Your job is to assume the role of a mental health therapist and provide sincere counselling advice to this person such that it improves the mental health of the person and alleviate the feelings of depression, sadness, or hopelessness from the person's mind.
You need to personalize your response by considering the specific conditions/concerns highlighted by the person. Following is an example:

{example}

The person expresses following internal feelings on social media:

{input}

Considering the feelings expressed by the person, you need to provide personalized counselling advice to this person by taking the role of 8 therapists dealing in different therapy treatment types: (1) Cognitive Behavioral Therapy (2) Exposure Therapy (3) Psychodynamic Therapy (4) Client Centered Therapy (5) Humanistic Therapy (6) Interpersonal Therapy (7) Mentalization Therapy (8) Mindfulness Therapy. Using each of the eight therapist personas, you firstly need to make a comprehensive personalized plan and then write a comprehensive personalized counselling advice as per the persons's feelings.
Moreover, both the plan and counselling should be comprehensive, personalized and written in an anonymous way i.e., the response should NOT mention the specific treatment type or therapist personas in the response. For example, you should not mention the phrases such as "cognitive behavioral therapy", "exposure therapy", "psychodynamic therapy", "client centered therapy", "humanistic therapy", "interpersonal therapy", "mentalization therapy", or "mindfulness therapy".
Your output should be of the following format:

Plan:
Cognitive Behavioral Therapist -> Your plan here by assuming the persona of therapist dealing in cognitive behavioral therapy
Exposure Therapist -> Your plan here by assuming the persona of therapist dealing in exposure therapy
Psychodynamic Therapist -> Your plan here by assuming the persona of therapist dealing in psychodynamic therapy
Client Centered Therapy -> Your plan here by assuming the persona of therapist dealing in client centered therapy
Humanistic Therapy -> Your plan here by assuming the persona of therapist dealing in Humanistic therapy
Interpersonal Therapy -> Your plan here by assuming the persona of therapist dealing in interpersonal therapy
Mentalization Therapy -> Your plan here by assuming the persona of therapist dealing in mentalization therapy
Mindfulness Therapy -> Your plan here by assuming the persona of therapist dealing in mindfulness therapy


Counsel:
Cognitive Behavioral Therapist -> Your counselling advice here by assuming the persona of therapist dealing in cognitive behavioral therapy
Exposure Therapist -> Your counselling advice here by assuming the persona of therapist dealing in exposure therapy
Psychodynamic Therapist -> Your counselling advice here by assuming the persona of therapist dealing in psychodynamic therapy
Client Centered Therapy -> Your counselling advice here by assuming the persona of therapist dealing in client centered therapy
Humanistic Therapy -> Your counselling advice here by assuming the persona of therapist dealing in Humanistic therapy
Interpersonal Therapy -> Your counselling advice here by assuming the persona of therapist dealing in interpersonal therapy
Mentalization Therapy -> Your counselling advice here by assuming the persona of therapist dealing in mentalization therapy
Mindfulness Therapy -> Your counselling advice here by assuming the persona of therapist dealing in mindfulness therapy
'''

cot_sdoh_prompt_dynamic = '''
{det_prompt}. Your job is to assume the role of a mental health therapist and provide sincere counselling advice to this person such that it improves the mental health state of the person and alleviate the feelings of depression, sadness, or hopelessness from the person's mind.
You need to personalize your response by considering the specific conditions/concerns highlighted by the person as well as the past history of the person. Following is an example:

{example}

The person expresses following internal feelings on social media:

{input}


{sdoh}

Considering the feelings expressed by the person on social media and the highlighted past history of the person, you need to provide personalized counselling advice to this person by taking the role of 8 therapists dealing in different therapy treatment types: (1) Cognitive Behavioral Therapy (2) Exposure Therapy (3) Psychodynamic Therapy (4) Client Centered Therapy (5) Humanistic Therapy (6) Interpersonal Therapy (7) Mentalization Therapy (8) Mindfulness Therapy. Using each of the eight therapist personas, you firstly need to make a comprehensive personalized plan and then write a comprehensive personalized counselling advice as per the persons's feelings and past history.
Moreover, both the plan and counselling should be comprehensive, personalized and written in an anonymous way i.e., the response should NOT mention the specific treatment type or therapist personas in the response. For example, you should not mention the phrases such as "cognitive behavioral therapy", "exposure therapy", "psychodynamic therapy", "client centered therapy", "humanistic therapy", "interpersonal therapy", "mentalization therapy", or "mindfulness therapy".
Your output should be of the following format:

Plan:
Cognitive Behavioral Therapist -> Your plan here by assuming the persona of therapist dealing in cognitive behavioral therapy
Exposure Therapist -> Your plan here by assuming the persona of therapist dealing in exposure therapy
Psychodynamic Therapist -> Your plan here by assuming the persona of therapist dealing in psychodynamic therapy
Client Centered Therapy -> Your plan here by assuming the persona of therapist dealing in client centered therapy
Humanistic Therapy -> Your plan here by assuming the persona of therapist dealing in Humanistic therapy
Interpersonal Therapy -> Your plan here by assuming the persona of therapist dealing in interpersonal therapy
Mentalization Therapy -> Your plan here by assuming the persona of therapist dealing in mentalization therapy
Mindfulness Therapy -> Your plan here by assuming the persona of therapist dealing in mindfulness therapy


Counsel:
Cognitive Behavioral Therapist -> Your counselling advice here by assuming the persona of therapist dealing in cognitive behavioral therapy
Exposure Therapist -> Your counselling advice here by assuming the persona of therapist dealing in exposure therapy
Psychodynamic Therapist -> Your counselling advice here by assuming the persona of therapist dealing in psychodynamic therapy
Client Centered Therapy -> Your counselling advice here by assuming the persona of therapist dealing in client centered therapy
Humanistic Therapy -> Your counselling advice here by assuming the persona of therapist dealing in Humanistic therapy
Interpersonal Therapy -> Your counselling advice here by assuming the persona of therapist dealing in interpersonal therapy
Mentalization Therapy -> Your counselling advice here by assuming the persona of therapist dealing in mentalization therapy
Mindfulness Therapy -> Your counselling advice here by assuming the persona of therapist dealing in mindfulness therapy
'''

# If you were the person who tweeted and you are given serveral choices, decide which choice is most promising to alleviate the suicidal thoughts
vote_prompt = '''Given a user input detailing an ill mental health issue and several choices providing relevant counselling, decide which choice is most promising to alleviate feelings of depression, sadness, or hopelessness from a user's mind suffering from ill mental health. Analyze each choice in detail and then assign a score rating (1-10) for 8 factors given below as used in psychotherapy or counselling.
Finally sum up the scores of each choice for all the 8 factors and then conclude in the last line "The best choice is (s)", where s the integer id of the choice having the maximum overall combined score.:

1. Medium Sensitivity (MS)
2. Hope and Positive Expectations (HPE)
3. Persuasiveness (PER)
4. Emotional Engagement (EEn)
5. Warmth, Acceptance & Understanding (WAU)
6. Empathy (EMP)
7. Alliance-Bond Capacity (ABC)
8. Alliance Rupture Repair Responsiveness (ARRR)

User Input: {input}

'''

compare_prompt = '''Briefly analyze the mental health counselling advice given in the following two passages to thwart the generation of suicidal thoughts in a person's mind. Conclude in the last line "The more suitable passage is 1", "The more suitable passage is 2", or "The two passages are equally suitable".
'''

score_prompt = '''Analyze the following passage in the context that it provides mental heatlh counselling advice to a person struggling with mental health problems, then at the last line conclude "Thus the suitablity score is {s}", where s is an integer from 1 to 10.
'''