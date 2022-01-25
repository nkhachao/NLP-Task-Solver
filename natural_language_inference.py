from pathlib import Path
import numpy as np
from models.natural_language_inference.discrete import predict_nli, label_types

project_root = str(Path(__file__).parent)


def preprocess(query):
    components = query.split('"')

    return components[1]


def classify_relationship(query, passage):
    hypothesis = preprocess(query)
    response = 'I think the statement \"' + hypothesis + '\" '

    predictions = predict_nli(passage, [hypothesis])[0]
    label = label_types[np.argmax(predictions)]

    if label == 'neutral':
        response += 'is neutral'
    elif label == 'contradiction':
        response += 'contradicts the passage'
    elif label == 'entailment':
        response += 'supports the passage'

    return response


if __name__ == "__main__":
    passage = '\'You were trouble'
    question = 'Is "You were not trouble" relevant to the passage?'

    print(classify_relationship(question, passage))