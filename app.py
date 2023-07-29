import streamlit as st

from modeling.model_handler import ModelHandler

MODELS = {
    'XGB': None,
    'BERT': None
}


def load_model(model_type):
    global MODELS
    if model_type == 'XGB':
        MODELS[model_type] = ModelHandler.load_model_skl('XGB_Chain')
    else:
        MODELS[model_type] = ModelHandler.load_model_hf('bert-genre-classifier')


def get_movie_genres(summary, model_type):
    global MODELS

    if model_type == 'XGB':
        clf, mlb = MODELS[model_type]
        res = ModelHandler.inference_model_skl(clf, mlb, summary, 0.1)
    else:
        clf, tok, mlb = MODELS[model_type]
        res = ModelHandler.inference_model_hf(clf, tok, mlb, summary, 0.1)

    return {genre: f'{int(proba * 100)}%' for genre, proba in res}


st.title("Movie Genre Predictor")
model_type = st.selectbox('Select Model Type', ['XGB', 'BERT'])

movie_summary = st.text_area("Enter a movie summary:", "")

if st.button("Predict"):
    if MODELS[model_type] is None:
        load_model(model_type)

    genres_probs = get_movie_genres(movie_summary, model_type)
    st.dataframe(genres_probs)
