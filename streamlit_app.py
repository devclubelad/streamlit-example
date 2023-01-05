import pandas as pd
import numpy as np
import streamlit as st
import openai
from openai.embeddings_utils import cosine_similarity
from openai.embeddings_utils import get_embedding


openai.api_key = "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"
df2 = pd.read_csv('text-data_with_embeddings.csv')
df2["ada_search"] = df2["ada_search"].apply(eval).apply(np.array)

def get_similar_questions(question, df, top_n=3):
    question_embedding = get_embedding(question, engine='text-embedding-ada-002')
    df['similarity'] = df.ada_search.apply(lambda x: cosine_similarity(x, question_embedding))
    return df.sort_values('similarity', ascending=False).head(top_n)

def get_similar_questions_pretty(question, df, top_n=3):
    values = get_similar_questions(question, df, top_n)
    return values.doc_title.to_numpy()

def get_similar_title(question):
    return get_similar_questions_pretty(question, df2)

def search_in_csv(search_term):
    if search_term == '':
        return ''
    return get_similar_questions_pretty(search_term, df2)


search_term = st.text_input("Please enter a search phrase:")
st.write('The results are:', search_in_csv(search_term))
