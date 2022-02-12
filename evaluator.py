import pandas as pd
import numpy as np
from models.natural_language_inference.joined import predict_nli, label_types
from question_answering import answer
from evaluation_resources import squad
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
    squad.evaluate(answer)


if __name__ == "__main__":
    eval_qa()