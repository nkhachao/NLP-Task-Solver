import pandas as pd
import numpy as np
from models.natural_language_inference.joined import predict_nli, label_types
from statistics import mean
from tqdm import tqdm


def eval_nli():
    points = []

    data = pd.read_csv('evaluation_resources/multinli-processed/eval_matched.csv')

    premises = data['premise']
    hypotheses = data['hypothesis']
    labels = data['label']

    for premise, hypothesis, label in tqdm(zip(premises, hypotheses, labels), total=len(premises)):
        predictions = predict_nli(premise, [hypothesis])[0]
        points.append(1 if np.argmax(predictions) == label_types.index(label) else 0)

    print(mean(points)*100)


def eval_qa():
    def normalize_text(s):
        """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
        import string, re

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_f1(prediction, truth):
        pred_tokens = normalize_text(prediction).split()
        truth_tokens = normalize_text(truth).split()

        # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        common_tokens = set(pred_tokens) & set(truth_tokens)

        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            return 0

        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)

        return 2 * (prec * rec) / (prec + rec)

    points = []


if __name__ == "__main__":
    eval_nli()