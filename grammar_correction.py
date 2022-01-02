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

    return sentences, merge_words


def construct_response(sentences, merge_words):
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

    fixed_sentences = []

    print(sentences)

    for sentence in sentences:
        input_ids = tokenize('grammar: ' + sentence)
        outputs = model.generate(input_ids)

        fixed_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        fixed_sentences.append(fixed_sentence)

        fixed_words += word_difference(tokenizer.decode(input_ids[0], skip_special_tokens=True), fixed_sentence)

    response = str(fixed_words) + ' words fixed.\n' + construct_response(fixed_sentences, merge_words)
    print(response)
    return response


if __name__ == "__main__":
    passage = 'I are Hao'

    print(correct_grammar(passage))