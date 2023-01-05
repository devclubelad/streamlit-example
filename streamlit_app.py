import pandas as pd
import numpy as np
from streamlit_chat import message
import streamlit as st
import openai
from openai.embeddings_utils import cosine_similarity
from openai.embeddings_utils import get_embedding

openai.api_key = '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'

### Chat GPT section
COMPLETIONS_MODEL = "text-davinci-003"
df = pd.read_csv('data_with_embeddings.csv')
df["ada_search"] = df["ada_search"].apply(eval).apply(np.array)


def get_similar_questions(question, df, top_n=3):
    question_embedding = get_embedding(question, engine='text-embedding-ada-002')
    df['similarity'] = df.ada_search.apply(lambda x: cosine_similarity(x, question_embedding))
    return df.sort_values('similarity', ascending=False).head(top_n)[df.similarity > 0.75]


def return_answer(question, df):
    similar_questions = get_similar_questions(question, df, top_n=5)
    text = ''
    # Iterate over the data frame rows
    if len(similar_questions) == 0:
        return "Apologies, No relevant results found"
    text = "Below are the top related places: \n"
    for index, row in similar_questions.iterrows():
        print(row.similarity)
        text += 'File: ' + str(row.file).removesuffix('.pdf') + ' Page: ' + str(row.page) + '\n; '

    return text


# Get the compbined column and combine to a single string
# def combine_text_from_column(df, column_name):
#     text = ""
#     for index, row in df.iterrows():
#         text += row[column_name] + "\n* "
#     return text

# def construct_prompt(question, df, top_n=2):
#     similar_questions = get_similar_questions(question, df, top_n)
#     context = combine_text_from_column(similar_questions, 'combined')
#     header =  header = """Answer the question based only on the provided context and nothing else, and if the answer is not contained within the context below, say "I don't know. Please contact our sales team: sales@venti.ai", do not invent or deduce!\n\n ====== \n\n Context:\n"""
#     return header + "".join(context) + """\n\n ======== \n\n Please answer in a prusasive and engaging way. \n\n Q: """ + question + "\n A:"


# COMPLETIONS_API_PARAMS = {
#     # We use temperature of 0.0 because it gives the most predictable, factual answer.
#     "temperature": 0.0,
#     "max_tokens": 300,
#     "model": COMPLETIONS_MODEL,
# }

# def answer_query_with_context(query: str, df: pd.DataFrame, show_prompt: bool = False) -> str:
#     long_prompt = construct_prompt(query, df)

#     # Add a first step for checking a question
#     prompt = """Below is an input, if it is a question write "true", otherwise  write "false" \n\n Input: """ + query + """\n\n"""

#     first_response = openai.Completion.create(prompt=prompt, **COMPLETIONS_API_PARAMS)
#     if "true" in first_response["choices"][0]["text"].lower():
#         # prompt = """ Below is a user input, if the user asked how to buy "true", if he asked anything else write "false" \n\n User Input:""" + query + """\n\n"""
#         # second_response = openai.Completion.create(prompt=prompt, **COMPLETIONS_API_PARAMS)
#         # if "true" in second_response["choices"][0]["text"].lower():
#             # return "This is great, You can purchase at venti.ai/purchase or contact our sales team: sales@venti.ai"
#         # else:
#             response = openai.Completion.create(prompt=long_prompt, **COMPLETIONS_API_PARAMS)
#     else:
#         prompt = """ Please answer the user input and encourage him to ask questions about Venti.ai. \n\n User Input: """ + query + """\n\n"""
#         # return openai.Completion.create(prompt=prompt, **COMPLETIONS_API_PARAMS)["choices"][0]["text"] + first_response["choices"][0]["text"].lower() + prompt + query +  "\n\nThis was not a question"
#         return openai.Completion.create(prompt=prompt, **COMPLETIONS_API_PARAMS)["choices"][0]["text"]

#     if show_prompt:
#         # print(prompt)
#         return long_prompt + '\n\n' + response["choices"][0]["text"] + "\n\n What else can I help you with?"
#     # return response["choices"][0]["text"].strip(" \n")
#     else:
#         return response["choices"][0]["text"] + "\n\n What else can I help you with?"

st.set_page_config(
    page_title="Level7 - Search App",
    page_icon=":robot:"
)

st.header("Level7 - Search Knowledge Base")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []


def get_text():
    input_text = st.text_input("You: ", "Search", key="input")
    return input_text


user_input = get_text()

if user_input:
    if user_input == "Search":
        output = "Hi, Please enter your search in the box above"
    else:

        # output = answer_query_with_context(user_input, df, show_prompt=True)
        output = return_answer(user_input, df)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
