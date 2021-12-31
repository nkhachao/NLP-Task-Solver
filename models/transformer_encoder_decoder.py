from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-small')

TOKEN_LIMIT = 192


def tokenize(text):
    if len(tokenizer.tokenize(text)) + 2 > TOKEN_LIMIT:
        return None

    tokenized = tokenizer(text, padding='max_length', max_length=TOKEN_LIMIT)
    return tokenized.input_ids, tokenized.attention_mask