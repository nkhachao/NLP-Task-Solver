from question_answering import answer
from named_entity_recognition import extract_named_entities
from grammar_correction import correct_grammar

def respond(passage, question):
    intent = classify_intent(question)

    if intent == 'NER':
        return extract_named_entities(passage)
    elif intent == 'GEC':
        return correct_grammar(passage)
    elif intent == 'QA':
        return answer(passage, question)


def classify_intent(question):
    request = question.lower()
    if 'entity' in request or 'entities' in request:
        return 'NER'
    if ('correct' in request or 'fix' in request) and ('passage' in request or 'sentence' in request):
        return 'GEC'
    else:
        return 'QA'