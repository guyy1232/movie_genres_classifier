import os
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, hamming_loss
from sklearn.pipeline import Pipeline
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast


class ModelHandler:
    @staticmethod
    def train_model_skl(clf, inputs, labels):
        clf_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
            ('clf', clf)
        ])
        # Train the classifier
        clf_pipeline.fit(inputs, labels)
        return clf_pipeline

    @staticmethod
    def evaluate_model_skl(clf_pipeline, mlb, test_input, test_labels):
        pred_proba = clf_pipeline.predict_proba(test_input)

        reports_by_tr = []
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            predictions = pred_proba > threshold

            report_str = f'Threshold: {threshold}\n' \
                         f'Humming loss: {hamming_loss(test_labels, predictions)}\n\n' \
                         f'{classification_report(test_labels, predictions, target_names=mlb.classes_, zero_division=False)}\n'
            reports_by_tr.append(report_str)
        return reports_by_tr

    @staticmethod
    def save_model_skl(clf_pipeline, mlb, evaluation_reports=None, model_path='classifier'):
        os.makedirs(model_path, exist_ok=True)
        dump(clf_pipeline, os.path.join(model_path, 'clf.joblib'))
        dump(mlb, os.path.join(model_path, 'mlb.joblib'))

        if evaluation_reports:
            with open(os.path.join(model_path, 'evaluation_report.txt'), 'w') as file:
                file.writelines(evaluation_reports)

    @staticmethod
    def load_model_skl(model_path):
        clf_pipeline = load(os.path.join(model_path, 'clf.joblib'))
        mlb = load(os.path.join(model_path, 'mlb.joblib'))

        return clf_pipeline, mlb

    @staticmethod
    def load_model_hf(model_path):
        model = BertForSequenceClassification.from_pretrained('guyyanko/bert-movie-genres-classification')
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        mlb = load(os.path.join(model_path, 'mlb.joblib'))
        return model, tokenizer, mlb

    @staticmethod
    def inference_model_skl(clf_pipeline, mlb, plot_summary, threshold=0.5):
        if type(plot_summary) == str:
            plot_summary = [plot_summary]

        probs = clf_pipeline.predict_proba(plot_summary)[0]

        labels = [(mlb.classes_[i], proba) for i, proba in enumerate(probs) if proba > threshold]

        return labels

    @staticmethod
    def inference_model_hf(model, tokenizer, mlb, plot_summary, threshold=0.5):
        model.to('cpu')

        inputs = tokenizer(plot_summary, truncation=True, padding=True, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.sigmoid(outputs.logits).numpy()[0]
        labels = [(mlb.classes_[i], proba) for i, proba in enumerate(probs) if proba > threshold]

        return labels
