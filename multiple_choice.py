from models.transformer_encoder_decoder import tokenizer
from transformers import T5ForConditionalGeneration
import numpy as np
from pathlib import Path

project_root = str(Path(__file__).parent)

model = T5ForConditionalGeneration.from_pretrained(project_root + '/models/multiple-choice/choice_picker')


def tokenize(text):
    return tokenizer(text, return_tensors='pt').input_ids


def preprocess(query):
    results = []
    text = []

    for word in query.split():
        mode = len(results)
        if mode == 0 and 'A.' in word or mode == 1 and 'B.' in word or mode == 2 and 'C.' in word or mode == 3 and 'D.' in word:
            results.append(' '.join(text))
            text = []
            continue

        text.append(word)

    results.append(' '.join(text))

    for i in range(len(results)):
        if results[i][-1] in [',']:
            results[i] = results[i][:-1]

    print(results)

    return results[0], results[1:]


def pick_choice(query, passage):
    question, choices = preprocess(query)

    choice0, choice1, choice2, choice3 = choices

    input_sequence = 'question: ' + question + ' choice 0: ' + choice0 + ' choice 1: ' + choice1 + ' choice 2: ' + choice2 + ' choice 3: ' + choice3 + ' passage: ' + passage

    input_ids = tokenize(input_sequence)

    output = model.generate(input_ids)[0]
    output = tokenizer.decode(output, skip_special_tokens=True)

    print(output)

    choice_names = ['A', 'B', 'C', 'D']

    if '0' in output:
        choice = 0
    elif '1' in output:
        choice = 1
    elif '2' in output:
        choice = 2
    elif '3' in output:
        choice = 3
    else:
        choice = 4

    return 'I choose ' + choice_names[choice] + ': ' + choices[choice]


if __name__ == "__main__":
    passage = 'John is my friend. Jill is not.'
    query = 'Who is my friend? A. John, B. Jill, C. Mary, D. Taylor'

    pick_choice(query, passage)