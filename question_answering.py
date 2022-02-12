from pathlib import Path
from models.question_answering.fine_tuned import answer_fine_tuned


project_root = str(Path(__file__).parent)


def preprocess(passage):
    passage = passage.replace('\r', '')
    passage = passage.replace('\n', ' ')
    passage = passage.replace('’', "'")
    passage = passage.replace('“', '"')
    passage = passage.replace('”', '"')
    passage = passage.replace('—', '-')

    return passage


def answer(question, passage):
    return answer_fine_tuned(question, preprocess(passage))

if __name__ == "__main__":
    passage = '\'Cause I knew you were trouble when you walked in. So shame on me now. Flew me to places I\'d never been \'Til you put me down, oh. I knew you were trouble when you walked in. So, shame on me now. Flew me to places I\'d never been. Now I\'m lyin\' on the cold hard ground'
    question = 'When did I know you were trouble?'

    print(answer(passage, question))