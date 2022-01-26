import torch
from torch import nn
from pathlib import Path
from transformers import AlbertTokenizer, AlbertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
project_root = str(Path(__file__).parent)

label_types = ['neutral', 'contradiction', 'entailment']


class Relevance(nn.Module):
    def __init__(self):
        super(Relevance, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text_embeddings):
        dropout_output = self.dropout(text_embeddings)
        linear_output = self.linear(dropout_output)

        return self.softmax(linear_output)


relevance = Relevance()
relevance.load_state_dict(torch.load(project_root + '/prediction_head/relevance', map_location=device))
bert = AlbertModel.from_pretrained(project_root + '/transformer')
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
relevance.eval()
bert.eval()


def tokenize(premise, hypothesis, max_length=512):
    if len(tokenizer(premise, hypothesis).input_ids) > max_length:
        return None

    tokenized = tokenizer(premise, hypothesis, return_tensors='pt')
    return tokenized.input_ids


def predict_nli(premise, hypotheses):
    results = []

    for hypothesis in hypotheses:
        _, text_embeddings = bert(input_ids=tokenize(premise, hypothesis), return_dict=False)

        relevance_score = relevance(text_embeddings)
        results.append(torch.squeeze(relevance_score).detach().numpy().tolist())

    return results
