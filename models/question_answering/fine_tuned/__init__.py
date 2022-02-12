import numpy as np
import torch
from torch import nn
from pathlib import Path
from transformers import AlbertTokenizer, AlbertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
project_root = str(Path(__file__).parent)

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TOKEN_LIMIT = 512


def tokenize(question, passage):
    tokenized = tokenizer(question, passage, padding='max_length', max_length=TOKEN_LIMIT)
    return tokenized.input_ids, tokenized.attention_mask


class SpanPredictor(nn.Module):
    def __init__(self):
        super(SpanPredictor, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.group = nn.Linear(768, 16)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.start_probabilities = nn.Linear(8192, 512)
        self.end_probabilities = nn.Linear(8192, 512)

    def forward(self, text_embeddings):
        dropouted = self.dropout(text_embeddings)
        reduced_embeddings = self.group(dropouted)
        relu_embeddings = self.relu(reduced_embeddings)
        flattened = self.flatten(relu_embeddings)
        start_probabilities = self.start_probabilities(flattened)
        end_probabilities = self.end_probabilities(flattened)

        return start_probabilities, end_probabilities


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = AlbertModel.from_pretrained(project_root + '/transformer')
        self.predictor = SpanPredictor()
        self.predictor.load_state_dict(torch.load(project_root + '/prediction_head/span_picker', map_location=device))

    def forward(self, text_ids, text_masks):
        text_embeddings, _ = self.bert(input_ids=text_ids, attention_mask=text_masks, return_dict=False)

        return self.predictor(text_embeddings)


model = Model().to(device)
model.eval()


def answer_span(input_ids, attention_mask):
    input_ids = torch.LongTensor([input_ids]).to(device)
    attention_mask = torch.LongTensor([attention_mask]).to(device)
    start_probabilities, end_probabilities = model(input_ids, attention_mask)
    start_probabilities = torch.squeeze(nn.Softmax(dim=1)(start_probabilities)).cpu().detach().numpy()
    end_probabilities = torch.squeeze(nn.Softmax(dim=1)(end_probabilities)).cpu().detach().numpy()

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


def answer_fine_tuned(question, passage):
    input_ids, attention_mask = tokenize(question, passage)

    if len(input_ids) <= TOKEN_LIMIT:
        start, end, start_probability, end_probability = answer_span(input_ids, attention_mask)

        return tokenizer.decode((input_ids[start:end + 1]))
    else:
        return ''


if __name__ == "__main__":
    passage = '\'Cause I knew you were trouble when you walked in. So shame on me now. Flew me to places I\'d never been \'Til you put me down, oh. I knew you were trouble when you walked in. So, shame on me now. Flew me to places I\'d never been. Now I\'m lyin\' on the cold hard ground'
    question = 'When did I know you were trouble?'

    print(answer_fine_tuned(passage, question))