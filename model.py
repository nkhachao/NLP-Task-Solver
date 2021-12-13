from transformers import pipeline

qa_pipeline = pipeline(
    "question-answering",
    model="csarron/bert-base-uncased-squad-v1",
    tokenizer="csarron/bert-base-uncased-squad-v1"
)


def answer(passage, question):
    predictions = qa_pipeline({'context': passage, 'question': question})
    return predictions['answer']


if __name__ == "__main__":
    passage = 'The iPhone is a line of smartphones designed and marketed by Apple Inc. that use Apple\'s iOS mobile operating system. The first-generation iPhone was announced by then-Apple CEO Steve Jobs on January 9, 2007. Since then, Apple has annually released new iPhone models and iOS updates. As of November 1, 2018, more than 2.2 billion iPhones had been sold.'
    question = 'When was the first-generation iPhone announced?'

    print(answer(passage, question))    # January 9, 2007