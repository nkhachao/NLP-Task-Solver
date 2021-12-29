from transformers import AlbertTokenizer, TFAlbertModel

TOKEN_LIMIT = 512

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
bert = TFAlbertModel.from_pretrained('albert-base-v2')


def tokenize(question, passage):
    return tokenizer(question, passage, padding='max_length', max_length=TOKEN_LIMIT, return_tensors="tf")


def encode(tokenized):
    outputs = bert(tokenized)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states[0]


def embed(question, passage):
    inputs = tokenize(question, passage)
    return encode(inputs)


def single_tokenize(sentence):
    return tokenizer(sentence, padding='max_length', max_length=TOKEN_LIMIT, return_tensors="tf")


def single_embed(sentence):
    inputs = single_tokenize(sentence)
    return encode(inputs)