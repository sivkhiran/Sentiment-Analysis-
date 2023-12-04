#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pickle
import numpy as np 
from PIL import Image

def load_model():
    with open ('processed_data.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()


def show_predict_page():
    st.title("NPR Speaker Identification with Text Data")
    original_title = '<p style="font-family:Serif; color:Black; font-size: 23px;">Daniel Thang,Reed Balentine,Khiran Kumar</p>'
    st.caption(original_title, unsafe_allow_html=True)
    st.header("Introduction")
    st.markdown('<div style="text-align: justify;"> It is a speaker identification project, where the creator uses dialogue from an adult cartoon show called “south park” \nto predict which character spoke that dialogue. They focused on the three main characters of the show and through \ndifferent classification models and RNN model (recurrent neural network), they evaluate the accuracy of the model’s prediction. We chose to analyze this project because \nthe objective of the project is quite similar to us as our project also deals with predicting the speaker from NPR \nMedia Dialog Transcripts. We also plan on using text classification models such as RNN, so analyzing this project would be beneficial to our \nproject’s success. Especially because the accuracy rates of the models from this project are quite low, so understanding the flaws and \nevaluating better methods and approaches for higher accuracy would allow us to also apply them into our project.</div>',unsafe_allow_html=True)
    st.subheader("Dataset")
    st.markdown("In the dataset we had total of 3,199,858 rows and 4 columns where 562 rows with missing utterance data, after the EDA attaching \nthe host-map there were a total of 1,174,823 rows and 6 columns.")
    url = 'https://www.kaggle.com/datasets/shuyangli94/interview-npr-media-dialog-transcripts'
    st.markdown(f'''
    <a href={url}><button style="background-color:white;">View Dataset Here</button></a>
    ''', unsafe_allow_html=True)
    st.subheader("Preprocessing")
    st.markdown ('<div style="text-align: justify;">Their dataset has 70,896 lines of dialogue by columns of season, episode, and character that spoke that line within the show. There were almost 3950 unique characters, but most of them only spoke a few times, so they had narrow it down to 3 main characters for the analysis to avoid imbalance classes of data.</div>',unsafe_allow_html=True)
    code = '''top_speakers = PreProcess_df.groupby(['speaker']).size().loc[PreProcess_df.groupby(['speaker']).size() > 10000]
           df = pd.DataFrame(PreProcess_df.loc[PreProcess_df['speaker'].isin(top_speakers.index.values)])
           df = df.reset_index(drop=True)'''
    st.code(code, language='python')
    pre_processing_code = '''def preprocess_text(text):
    tokeniser = RegexpTokenizer(r'\w+')
    tokens = tokeniser.tokenize(text)

    lemmatiser = WordNetLemmatizer()
    lemmas = [lemmatiser.lemmatize(token.lower(), pos='v') for token in tokens]

    keywords= [lemma for lemma in lemmas if lemma not in stopwords.words('english')]
    return keywords'''
    st.code(pre_processing_code, language='python')
    st.markdown ("Using the above code we found **Neal Connan**,**Steve Inskeep**,**Robert Siegel**,**Ira Flatow** are the top speakers")
    image = Image.open('project.png')
    st.image(image, width=700)
    st.subheader("Creating Pipelines")
    pipeline_code = '''pipe = make_pipeline(TfidfVectorizer(), MultinomialNB())
    pipe.steps

    param_grid = {}
    param_grid["tfidfvectorizer__max_features"] = [500, 1000, 15000]
    param_grid["tfidfvectorizer__ngram_range"] = [(1,1), (1,2), (2,2)]
    param_grid["tfidfvectorizer__stop_words"] = ["english", None]
    param_grid["tfidfvectorizer__strip_accents"] = ["ascii", "unicode", None]
    param_grid["tfidfvectorizer__analyzer"] = ["word", "char"]
    param_grid["tfidfvectorizer__binary"] = [True, False]
    param_grid["tfidfvectorizer__norm"] = ["l1", "l2", None]
    param_grid["tfidfvectorizer__use_idf"] = [True, False]
    param_grid["tfidfvectorizer__smooth_idf"] = [True, False]
    param_grid["tfidfvectorizer__sublinear_tf"] = [True, False]


    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', verbose = 3, n_jobs = -1)
    grid.get_params().keys()'''
    st.code(pipeline_code, language='python')


    speaker = (
        "Neal Connan",
        "Steve Inskeep",
        "Robert Siegel",
        "Ira Flatow",
    )

    episodes = (
        "57264",
        "58225",
        "68175",
        "70039",
        "58131",
        "74806",
        "79679",
        "87639",
        "86452",
    )

