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
        self.linear = nn.Linear(1536, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, passage_embeddings, label_embeddings):
        concatenated = torch.cat((passage_embeddings, label_embeddings), dim=1)
        dropout_output = self.dropout(concatenated)
        linear_output = self.linear(dropout_output)

        return self.softmax(linear_output)


relevance = Relevance()
relevance.load_state_dict(torch.load(project_root + '/prediction_head/relevance', map_location=device))
bert = AlbertModel.from_pretrained(project_root + '/transformer')
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
relevance.eval()
bert.eval()


def tokenize(text):
    if len(tokenizer.tokenize(text)) + 2 > 512:
        return None

    tokenized = tokenizer(text, return_tensors='pt')
    return tokenized.input_ids


def predict_nli(premise, hypotheses):
    results = []
    _, premise_embeddings = bert(input_ids=tokenize(premise), return_dict=False)

    for hypothesis in hypotheses:
        _, hypothesis_embeddings = bert(input_ids=tokenize(hypothesis), return_dict=False)

        relevance_score = relevance(premise_embeddings, hypothesis_embeddings)
        results.append(torch.squeeze(relevance_score).detach().numpy().tolist())

    return results
