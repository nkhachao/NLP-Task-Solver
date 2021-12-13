from tensorflow import keras
import numpy as np
from transformers import AlbertTokenizer, TFAlbertModel

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
bert = TFAlbertModel.from_pretrained('albert-base-v2')

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


start_classifier = keras.models.load_model('models/start_probability')
end_classifier = keras.models.load_model('models/end_probability')


def answer(passage, question):
    input_ids = tokenize(question, passage).input_ids[0].numpy()

    embeddings = embed(question, passage)
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

    #print('start:', start, ', confidence:', start_probability, '| end:', end, ', confidence:', end_probability)
    return tokenizer.decode((input_ids[start:end+1]).tolist())


if __name__ == "__main__":
    passage = 'In information retrieval, tf–idf, TF*IDF, or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.[1] It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling. The tf–idf value increases proportionally to the number of times a word appears in the document and is offset by the number of documents in the corpus that contain the word, which helps to adjust for the fact that some words appear more frequently in general. tf–idf is one of the most popular term-weighting schemes today. A survey conducted in 2015 showed that 83% of text-based recommender systems in digital libraries use tf–idf.'
    question = 'What is tf-idf often used at?'

    print(answer(passage, question))    # January 9, 2007