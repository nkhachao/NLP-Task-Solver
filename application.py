from re import template
import re
from flask import Flask, render_template, request, redirect, url_for
from tensorflow import keras
import numpy as np
from transformers import AlbertTokenizer, TFAlbertModel
from pathlib import Path
from responder import respond

project_root = str(Path(__file__).parent)

app = Flask(__name__, template_folder='template')

# passage = 'The iPhone is a line of smartphones designed and marketed by Apple Inc. that use Apple\'s iOS mobile operating system. The first-generation iPhone was announced by then-Apple CEO Steve Jobs on January 9, 2007. Since then, Apple has annually released new iPhone models and iOS updates. As of November 1, 2018, more than 2.2 billion iPhones had been sold.'
# question = 'When was the first-generation iPhone announced?'

# passage = '\'Cause I knew you were trouble when you walked in. So shame on me now. Flew me to places I\'d never been \'Til you put me down, oh. I knew you were trouble when you walked in. So, shame on me now. Flew me to places I\'d never been. Now I\'m lyin\' on the cold hard ground'
#     question = 'When did I know you were trouble?'



@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template("home.html")

@app.route('/get')
def get_bot_respone():
    if request.args.get('psg') == "":
        return "Please enter the passage!"
    else:
        return respond(str(request.args.get('psg')), str(request.args.get('msg')))

if __name__ == "__main__":
    app.run()