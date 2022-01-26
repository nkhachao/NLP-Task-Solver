from pathlib import Path
import numpy as np
from models.natural_language_inference.discrete import predict_nli, label_types

project_root = str(Path(__file__).parent)


def preprocess(query):
    components = query.split('"')

    hypotheses = []
    for i in range(1, len(components)):
        if i%2 == 0:
            continue
        hypotheses.append(components[i])

    print(hypotheses)

    return hypotheses


# 'neutral', 'contradiction', 'entailment'

def pick_class(query, passage):
    hypotheses = preprocess(query)

    predictions = predict_nli(passage, hypotheses)
    print(predictions)

    best_relevance = -10
    best_index = -1
    for i in range(len(predictions)):
        relevance_score = predictions[i][2] - predictions[i][1]
        print(relevance_score)

        if relevance_score > best_relevance:
                best_relevance = relevance_score
                best_index = i

    response = 'I think the passage belongs to the class \"' + hypotheses[best_index] + '\" '

    return response


if __name__ == "__main__":
    passage = '\'The iPhone was introduced in 2007'
    question = 'Which class does the passage belong to? "Technology", "Politics", "Sport", "Business"'

    print(pick_class(question, passage))