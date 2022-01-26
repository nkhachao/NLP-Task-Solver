from model_manager import fetch_models
from flask import Flask, render_template, request, redirect, url_for
import flask
from pathlib import Path
from responder import respond
import threading

app = Flask(__name__, template_folder='template')




@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template("home.html")

@app.route('/get')
def get_bot_respone():
    if request.args.get('psg') == "":
        return "Please enter the passage!"
    else:
        processed_response = ''
        response = respond(str(request.args.get('psg')), str(request.args.get('msg')))

        #t1 = threading.Thread(target=speak, args=(response,))
        #t1.start()
        #t1.join()

        for line in response.split('\n'):
            processed_response += flask.Markup.escape(line) + flask.Markup('<br />')
        return processed_response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)