from models.transformer_encoder_decoder import tokenizer
from transformers import T5ForConditionalGeneration
import numpy as np
from pathlib import Path
import torch

project_root = str(Path(__file__).parent)

model = T5ForConditionalGeneration.from_pretrained(project_root + '/models/grammar-correction/grammar_corrector')


def tokenize(text):
    return tokenizer(text, return_tensors='pt').input_ids


def preprocess(passage):
    sentences = []
    merge_words = []

    passage = passage.replace('\r', '')
    paragraphs = passage.split('\n')
    for paragraph in paragraphs:
        paragraph_sentences = paragraph.split('. ')

        sentences += paragraph_sentences
        merge_words += [' '] * (len(paragraph_sentences) - 1)

        merge_words.append('\n')

    merge_words = merge_words[0:len(sentences) - 1]

    for i in range(len(sentences)):
        if sentences[i][-1] not in ['?', '!', '.']:
            sentences[i] = sentences[i] + '.'

    return sentences, merge_words


def recontruct_passage(sentences, merge_words):
    response = sentences[0]

    for sentence, merge_word in zip(sentences[1:], merge_words):
        response += merge_word + sentence

    return response


def word_difference(a, b):
    words_a = a.split(' ')
    words_b = b.split(' ')

    difference = 0

    if len(words_a) > len(words_b):
        long_list = words_a
        short_list = words_b
    else:
        long_list = words_b
        short_list = words_a

    for word in long_list:
        if word not in short_list:
            difference += 1

    return difference


def correct_grammar(passage):
    fixed_words = 0
    sentences, merge_words = preprocess(passage)

    modified_sentences = []
    checked_sentences = []

    print(sentences)

    for sentence in sentences:
        input_ids = tokenize('grammar: ' + sentence)
        outputs = model.generate(input_ids)

        checked_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        checked_sentences.append(checked_sentence)

        fixed_words += word_difference(sentence, checked_sentence)

        if fixed_words > 0:
            modified_sentences.append((sentence, checked_sentence))

    fixed_passage = recontruct_passage(checked_sentences, merge_words)
    response = str(fixed_words) + ' words fixed.\n\n'

    for faulty_sentence, fixed_sentence in modified_sentences:
        response += '"' + faulty_sentence + '" -> "' + fixed_sentence + '"\n'

    response += '\nFixed passage:\n' + fixed_passage

    print(response)
    return response


if __name__ == "__main__":
    passage = 'The iPhone is a line of smartphones designed and marketed by Apple Inc. that use Apple\'s iOS mobile operating system. The first-generation iPhone was announced by then-Apple CEO Steve Jobs on January 9, 2007. Since then, Apple has annually released new iPhone models and iOS updates. As of November 1, 2018, more than 2.2 billion iPhones had been sold.'

    correct_grammar(passage)