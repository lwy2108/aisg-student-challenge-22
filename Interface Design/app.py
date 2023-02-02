import flask
import pickle

with open(f'emotion_classifier.pkl','rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':

        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        sentText = flask.request.form['text']
        
        input_variables = sentText
        prediction = model.predict([input_variables])
        return flask.render_template('main.html',
                                     original_input=sentText,
                                     result=prediction[0]
                                     )
if __name__ == '__main__':
    app.run(port=8090,debug=True)
