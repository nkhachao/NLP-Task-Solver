from re import template
import re
from flask import Flask, render_template, request, redirect, url_for
import datetime
from responder import respond

app = Flask(__name__, template_folder='template')


@app.route('/', methods=['POST', 'GET'])
def index():
    ans =""
    if request.method == 'POST':
        if request.form['button'] == "Send":
            ans = [request.form['passage'], request.form['question'], respond(request.form['passage'], request.form['question'])]

    return render_template("home.html", data = ans)

if __name__ == "__main__":
    app.run()