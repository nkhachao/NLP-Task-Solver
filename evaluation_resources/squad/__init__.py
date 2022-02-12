"""Modified Official evaluation script for SQuAD version 2.0.

In addition to basic functionality, we also compute additional statistics and
plot precision-recall curves if an additional na_prob.json file is provided.
This file is expected to map question ID's to the model's predicted probability
that a question is unanswerable.
"""
from collections import Counter
import re
import string
import json
from pathlib import Path
from tqdm import tqdm

project_root = str(Path(__file__).parent)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(model):
    f = open(project_root + '/dev-v1.1.json')
    dataset = json.load(f)['data']

    f1 = total = 0
    for article in tqdm(dataset):
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                passage = paragraph['context']
                question = qa['question']
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = model(question, passage)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    f1 = 100.0 * f1 / total

    print(f1)



if __name__ == '__main__':
    evaluate(None)
