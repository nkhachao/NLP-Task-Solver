from re import template
import re
from flask import Flask, render_template, request, redirect, url_for
import datetime
from transformers import pipeline

app = Flask(__name__, template_folder='template')

# passage = 'The iPhone is a line of smartphones designed and marketed by Apple Inc. that use Apple\'s iOS mobile operating system. The first-generation iPhone was announced by then-Apple CEO Steve Jobs on January 9, 2007. Since then, Apple has annually released new iPhone models and iOS updates. As of November 1, 2018, more than 2.2 billion iPhones had been sold.'
# question = 'When was the first-generation iPhone announced?'

qa_pipeline = pipeline(
    "question-answering",
    model="csarron/bert-base-uncased-squad-v1",
    tokenizer="csarron/bert-base-uncased-squad-v1"
)

def answer(passage, question):
    predictions = qa_pipeline({'context': passage, 'question': question})
    return predictions['answer']

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template("home.html")

@app.route('/get')
def get_bot_respone():
    if request.args.get('psg') == "":
        return "Please enter the passage!"
    else:

        return answer(str(request.args.get('psg')), str(request.args.get('msg')))

if __name__ == "__main__":
    app.run()