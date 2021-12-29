from models.transformer_encoder_decoder import tokenize, tokenizer
from transformers import T5ForConditionalGeneration
import numpy as np
from pathlib import Path
import torch

project_root = str(Path(__file__).parent)

model = T5ForConditionalGeneration.from_pretrained(project_root + '/models/grammar-correction/grammar_corrector')


def correct_grammar(text):
    input_ids, attention_mask = tokenize('grammar: ' + text)  # Batch size 1
    outputs = model.generate(torch.LongTensor(([input_ids])), attention_mask=torch.LongTensor(([attention_mask])))

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    sentence = 'My name are Hao'

    print(correct_grammar(sentence))