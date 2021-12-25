from question_answering import answer
from named_entity_recognition import extract_named_entities

def respond(passage, question):
    if 'entity' in question or 'entities' in question:
        return extract_named_entities(passage)
    else:
        return answer(passage, question)