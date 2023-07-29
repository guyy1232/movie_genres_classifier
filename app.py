import streamlit as st

from modeling.model_handler import ModelHandler

CLASSIFIER = None
TOKENIZER = None
MLB = None


def load_model(model_type):
    global CLASSIFIER, TOKENIZER, MLB
    if model_type == 'XGB':
        CLASSIFIER, MLB = ModelHandler.load_model_skl('XGB_Chain')
    else:
        CLASSIFIER, TOKENIZER, MLB = ModelHandler.load_model_hf('bert-genre-classifier')


def get_movie_genres(summary, model_type):
    global CLASSIFIER, TOKENIZER, MLB

    if model_type == 'XGB':
        res = ModelHandler.inference_model_skl(CLASSIFIER, MLB, summary, 0.1)
    else:
        res = ModelHandler.inference_model_hf(CLASSIFIER, TOKENIZER, MLB, summary, 0.1)

    return {genre: f'{int(proba * 100)}%' for genre, proba in res}


st.title("Movie Genre Predictor")
model_type = st.selectbox('Select Model Type', ['XGB', 'BERT'])

movie_summary = st.text_area("Enter a movie summary:", "")

if st.button("Predict"):
    load_model(model_type)

    genres_probs = get_movie_genres(movie_summary, model_type)
    st.dataframe(genres_probs)
