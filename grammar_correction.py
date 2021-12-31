from models.transformer_encoder_decoder import tokenize, tokenizer
from transformers import T5ForConditionalGeneration
import numpy as np
from pathlib import Path
import torch

project_root = str(Path(__file__).parent)

model = T5ForConditionalGeneration.from_pretrained(project_root + '/models/grammar-correction/grammar_corrector')


def preprocess(passage):
    sentences = []
    merge_words = []

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

    print(response)
    return response


def correct_grammar(passage):
    sentences, merge_words = preprocess(passage)

    fixed_sentences = []

    for sentence in sentences:
        input_ids, attention_mask = tokenize('grammar: ' + sentence)
        outputs = model.generate(torch.LongTensor(([input_ids])), attention_mask=torch.LongTensor(([attention_mask])))

        fixed_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        fixed_sentences.append(fixed_sentence)

    return construct_response(fixed_sentences, merge_words)


if __name__ == "__main__":
    passage = 'I are Hao'

    print(correct_grammar(passage))