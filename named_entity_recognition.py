from models.transformer_encoder import tokenizer, single_embed
from pathlib import Path
from tensorflow import keras
import numpy as np

project_root = str(Path(__file__).parent)

ner_tagger = keras.models.load_model(project_root + '/models/named-entity-recognition/ner_tagger')

tag_types = ['I-LOC', 'B-ORG', 'O', 'B-PER', 'I-PER', 'I-MISC', 'B-MISC', 'I-ORG', 'B-LOC']
tag_translation = ['Location', 'Organization', '', 'Person', 'Person', 'Other', 'Other', 'Organization', 'Location']


def construct_response(entity_names, tag_names):
    response = 'There are ' + str(len(entity_names)) + ' named entities in this passage.\n\n'
    for entity_name, tag_name in zip(entity_names, tag_names):
        response += '"' + entity_name + '": ' + tag_name + '\n'

    return response[0:-1]

def extract_named_entities(sentence):
    embeddings = single_embed(sentence)
    words = sentence.split(' ')
    word_tokens = [tokenizer.tokenize(x) for x in words]

    predictions = ner_tagger(np.array([embeddings]))[0]

    entities = []
    tags = []

    i = 1
    for word_index in range(len(word_tokens)):
        tag = None
        tagged_token_length = 0

        for token in word_tokens[word_index]:
            if predictions[i] != 2:
                if len(token) > tagged_token_length:
                    tag = predictions[i]
                    tagged_token_length = len(token)

            i += 1

        if tag:
            entities.append(word_index)
            tags.append(tag)

    full_entity_indices = []
    tag_name = ''

    tag_names = []
    entity_names = []

    for word_index, tag in zip(entities, tags):
        if tag_name:
            if word_index - full_entity_indices[-1] == 1 and tag_translation[tag] == tag_name:
                full_entity_indices.append(word_index)
            else:
                entity = ' '.join([words[x] for x in full_entity_indices])
                entity_names.append(entity)
                tag_names.append(tag_name)

                full_entity_indices = [word_index]
                tag_name = tag_translation[tag]

        else:
            full_entity_indices = [word_index]
            tag_name = tag_translation[tag]

    if tag_name:
        entity = ' '.join([words[x] for x in full_entity_indices])
        entity_names.append(entity)
        tag_names.append(tag_name)

    return construct_response(entity_names, tag_names)


if __name__ == "__main__":
    sentence = 'The first-generation iPhone was announced by then-Apple CEO Steve Jobs on January 9, 2007.'

    print(extract_named_entities(sentence))