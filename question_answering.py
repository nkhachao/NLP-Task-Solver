from tensorflow import keras
import numpy as np
from transformers import AlbertTokenizer, TFAlbertModel
from pathlib import Path


project_root = str(Path(__file__).parent)


tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
bert = TFAlbertModel.from_pretrained('albert-base-v2')
start_classifier = keras.models.load_model(project_root + '/models/question-answering/start_probability')
end_classifier = keras.models.load_model(project_root + '/models/question-answering/end_probability')
token_classifier = keras.models.load_model(project_root + '/models/question-answering/classifier')

TOKEN_LIMIT = 512


def tokenize(question, passage):
    return tokenizer(question, passage, padding='max_length', max_length=TOKEN_LIMIT, return_tensors="tf")


def encode(tokenized):
    outputs = bert(tokenized)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states[0]


def embed(question, passage):
    inputs = tokenize(question, passage)
    return encode(inputs)


def split_passage(question, passage):
    sentences = passage.split('.')

    parts = []
    part = ''
    for sentence in sentences:
        if part == '':
            part = sentence
        else:
            if len(tokenize(question, part + '.' + sentence).input_ids[0]) <= 512:
                part += '.' + sentence
            else:
                parts.append(part)
                part = sentence

    parts.append(part)

    return parts


def preprocess(passage):
    passage = passage.replace('\r', '')
    passage = passage.replace('\n', ' ')
    passage = passage.replace('’', "'")
    passage = passage.replace('“', '"')
    passage = passage.replace('”', '"')
    passage = passage.replace('—', '-')

    return passage

def answer(passage, question):
    passage = preprocess(passage)
    input_ids = tokenize(question, passage).input_ids[0].numpy()

    if len(input_ids) <= TOKEN_LIMIT:
        embeddings = embed(question, passage)

        start, end, start_probability, end_probability = answer_span(embeddings)
        spans, confidence_scores = answer_tokens(embeddings)
        print('Model 1:', tokenizer.decode((input_ids[start:end+1]).tolist()), start_probability, end_probability, (start_probability + end_probability)/2, end - start)
        for span, confidence in zip(spans, confidence_scores):
            print('Model 2:', tokenizer.decode((input_ids[span]).tolist()), confidence)

        if (end - start > 23 and (start_probability + end_probability)/2 < 0.7) or end - start > 25:
            if spans:
                if np.max(confidence_scores) > 1.5:
                    return tokenizer.decode((input_ids[spans[np.argmax(confidence_scores)]]).tolist())

        return tokenizer.decode((input_ids[start:end+1]).tolist())
    else:
        paragraphs = split_passage(question, passage)
        for paragraph in paragraphs:
            input_ids = tokenize(question, paragraph).input_ids[0].numpy()

            print(paragraph)
            embeddings = embed(question, paragraph)

            start, end, start_probability, end_probability = answer_span(embeddings)

            print('Answer: ', tokenizer.decode((input_ids[start:end + 1]).tolist()))
            print('-----')



def answer_span(embeddings):
    start_probabilities = np.squeeze(start_classifier(np.array([embeddings])))
    end_probabilities = np.squeeze(end_classifier(np.array([embeddings])))

    sorted_start_probabilities = np.sort(start_probabilities)
    sorted_end_probabilities = np.sort(end_probabilities)

    start_probability = sorted_start_probabilities[-1]
    end_probability = sorted_end_probabilities[-1]

    sorted_start_probabilities = np.delete(sorted_start_probabilities, -1)
    sorted_end_probabilities = np.delete(sorted_end_probabilities, -1)

    start = np.where(start_probabilities == start_probability)[0][0]
    end = np.where(end_probabilities == end_probability)[0][0]

    while start > end:
        start_probability = sorted_start_probabilities[-1]
        end_probability = sorted_end_probabilities[-1]

        sorted_start_probabilities = np.delete(sorted_start_probabilities, -1)
        sorted_end_probabilities = np.delete(sorted_end_probabilities, -1)

        if start_probability > end_probability:
            start = np.where(start_probabilities == start_probability)[0][0]
            refreshed = 'start'
        else:
            end = np.where(end_probabilities == end_probability)[0][0]
            refreshed = 'end'

        if start > end:
            if refreshed == start:
                end = np.where(end_probabilities == end_probability)[0][0]
            else:
                start = np.where(start_probabilities == start_probability)[0][0]

    return start, end, start_probability, end_probability


def answer_tokens(embeddings):
    token_probabilities = np.squeeze(token_classifier(np.array([embeddings])))
    chosen_tokens = token_probabilities > 0.1

    indices = np.arange(512)
    chosen_indices = indices[chosen_tokens]

    span = []
    spans = []
    for index in chosen_indices:
        if not span:
            span.append(index)
        else:
            if index - span[-1] == 1:
                span.append(index)
            else:
                spans.append(span)
                span = [index]

    spans.append(span)
    confidence_scores = list(map(lambda x: sum(token_probabilities[x]), spans))

    return spans, confidence_scores

if __name__ == "__main__":
    passage = '\'Cause I knew you were trouble when you walked in. So shame on me now. Flew me to places I\'d never been \'Til you put me down, oh. I knew you were trouble when you walked in. So, shame on me now. Flew me to places I\'d never been. Now I\'m lyin\' on the cold hard ground'
    question = 'When did I know you were trouble?'

    print(answer(passage, question))