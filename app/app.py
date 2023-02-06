from flask import Flask, render_template, request, flash, redirect, url_for

import emoclass
import spanex

app = Flask(__name__)

app.secret_key = "12345"


@app.route('/')
def home():
    return render_template('causal.html')


@app.route('/', methods=['POST'])
def submit():
    conversation = request.form['input']
    if conversation:
        conversation = conversation.splitlines()
        senders, conversation = emoclass.split_sender(conversation)
        emotion, conversation = emoclass.classify(conversation)

        res = spanex.run(emotion, conversation.copy())

        return render_template('result.html', senders=senders, conversation=conversation, emotion=emotion, res=res)
    return redirect(url_for('home'))


@app.route('/result')
def result():
    return render_template('result.html')


@app.route('/result', methods=['GET'])
def back():
    return render_template('causal.html')


if __name__ == '__main__':
    app.run()
