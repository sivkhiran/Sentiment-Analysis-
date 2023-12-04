import streamlit as st
import pandas as pd
import pickle
import numpy as np 
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
def load_model():
    with open ('processed_data.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()


def show_predict_page():
    st.title("NPR Speaker Identification with Text Data")
    st.sidebar.markdown("# Main page")
    original_title = '<p style="font-family:Serif; color:Black; font-size: 23px;">Daniel Thang, Reed Balentine, Khiran Kumar Chidambaram Sivaraman</p>'
    st.caption(original_title, unsafe_allow_html=True)
    st.header("Introduction")
    st.markdown('<div style="text-align: justify;"> It is a speaker identification project, where the creator uses dialogue from an adult cartoon show called “south park” \nto predict which character spoke that dialogue. They focused on the three main characters of the show and through \ndifferent classification models and RNN model (recurrent neural network), they evaluate the accuracy of the model’s prediction. We chose to analyze this project because \nthe objective of the project is quite similar to us as our project also deals with predicting the speaker from NPR \nMedia Dialog Transcripts. We also plan on using text classification models such as RNN, so analyzing this project would be beneficial to our \nproject’s success. Especially because the accuracy rates of the models from this project are quite low, so understanding the flaws and \nevaluating better methods and approaches for higher accuracy would allow us to also apply them into our project.</div>',unsafe_allow_html=True)
    st.subheader("Dataset")
    st.markdown("In the dataset we had total of 3,199,858 rows and 4 columns where 562 rows with missing utterance data, after the EDA attaching \nthe host-map there were a total of 1,174,823 rows and 6 columns.")
    url = 'https://www.kaggle.com/datasets/shuyangli94/interview-npr-media-dialog-transcripts'
    st.markdown(f'''
    <a href={url}><button style="background-color:white;">View Dataset Here</button></a>
    ''', unsafe_allow_html=True)
    st.subheader("EDA and Data Cleaning")
    st.markdown("- All string lower case") 
    st.markdown("- Map with host-file (file that has NPR host name and host_id)")
    st.markdown("- Only keep rows with more than or equal to 100 characters in the utterance/line.")
    st.markdown("- Only keep top the 5 host/speaker")
    st.markdown("- Drop any rows if the string in the utterance column starts with “undefined”.")
    st.markdown("- Use SampleBy() method to limit the counts of host “Neal Conan” due to it being too high compared to other hosts. (Balance dataset)")
    st.markdown("- Drop all Nans")
    st.markdown("- Drop all columns except the speaker and utterance column")
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
    st.subheader("Random Forest and MultinomialNB")
    randomforest_code = '''from sklearn.ensemble import RandomForestClassifier
Tfid_rf_pipeline = Pipeline([
('Tfid', Tfid_Vector),
('Random_Forest', RandomForestClassifier())])
rf_gs = GridSearchCV(Tfid_rf_pipeline, rf_parameters, cv=3, verbose = 3, n_jobs = 1, scoring='accuracy')
print(rf_gs.best_params_)
print(rf_gs.best_score_)'''
    st.code(randomforest_code, language ='python')
    image = Image.open('Rf.png')
    st.image(image, width=700)
    multinomialNB_code ='''print("Score of model is :" , MultinomialNB_gs.score(X_test, y_test))
print("Precision Score : ", precision_score(y_test, test_pred, average='weighted'))
print("Recall Score : ", recall_score(y_test, test_pred, average='weighted'))
print('F1 value : ', f1_score(y_test, test_pred,average='weighted'))'''
    st.code(multinomialNB_code, language = 'python')
    image = Image.open('MNNB.png')
    st.image(image, width=700)                         
    st.subheader("Modelling and Evaluating")
    st.markdown('<div style="text-align: justify;"> We have used common parameters and used Multinomial Naive Bayes. It is useful for classification with discreate features, so here in text learning, the count of each word is used to predict the class or label. We got a model accuracy of 0.61, and I think this is good for starting point as we did not even use the best parameters for modeling just yet. For evaluation, we tested out the model by allowing it to predict who the speaker is based on the given text, and it was both correct for the two test. </div>',unsafe_allow_html=True)
    evaluation_code = '''vector = TfidfVectorizer(analyzer='word', stop_words='english', max_features = 850, ngram_range=(1, 1), 
                       binary=False, norm=None, smooth_idf=True, strip_accents=None,
                       sublinear_tf=True, use_idf=False)

    df_transformed = vector.fit_transform(X_train)
    multi_nb_model = MultinomialNB()
    multi_nb_model.fit(df_transformed, y_train)
    print ("Model accuracy within dataset: ", multi_nb_model.score(df_transformed, y_train))
    test_text = ["So for a top competitor like Lance to try to"]
    test_text_transform = vector.transform(test_text)

    print (multi_nb_model.predict(test_text_transform)," most likely said it.")'''
    st.code(evaluation_code, language='python')
    st.subheader("Feature Extraction and Transformers pipelines")
    st.markdown("- For feature transformers, the open-source NLP python library, NLTK, is mainly used. The methods include, regex tokenizer, which basically splits text into words or sub-words, English stop word remover, and wordnetlemmatizer, which provides semantic relationships between its words.")
    st.markdown("- For feature extractors, TFIDF vectorizer is used to give numerical representations of words and provide many other preprocessing methods that we might have missed.")
    st.subheader("Failed Models")
    st.markdown("We Also tried BERT and Multilayer Perceptron model")  
    st.markdown("Feature Extractors: For vectors, we tried TF-IDF and Word2Vec, but they got a relatively low model accuracy rate compared to CountVectorizer, so they were removed.")
    st.markdown("Feature Transformers: Normalizer and standard scaler were used but they did not have much impact as in one run, they decreased the model accuracy by around 0.5%, so they were removed.")
    st.subheader("BERT")
    st.markdown("We dedicated one notebook to the BERT model, as the processing methods are much different from other models. BERT itself already has a lot of the inbuilt cleaning and preprocessing methods, so we did not have to do much cleaning other than balancing the dataset as it is a necessary component of running a BERT model. We used a pre-trained BERT model imported from sckit-learn, and it took many hours to run in a local host, but only gave around 14-15% accuracy rate. So, we decided to not use the notebook, but added it to ourt github repo to display our project’s progress.")
    st.subheader("Multilayer Perceptron Model")
    st.markdown("The multilayer perceptron model did have the potential to reach a decent accuracy model rate as it is a Neural Network model. However, this was the last model to be used, so due to time constraints, it was not smart to invest time as the accuracy started out quite low as 20%. It would be more beneficial to focus on increasing the model accuracy of our main models that displayed great accuracy from the start.")
    st.subheader("Conclusion")
    image = Image.open('Screenshot (172).png')
    st.image(image, width=700) 

page_names_to_funcs = {
    "Main Page": show_predict_page,
    
}


    
    
    
