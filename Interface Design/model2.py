import flask
import pickle
from statistics import mode

from sgnlp.models.emotion_entailment import (
    RecconEmotionEntailmentConfig,
    RecconEmotionEntailmentTokenizer,
    RecconEmotionEntailmentModel,
    RecconEmotionEntailmentPreprocessor,
    RecconEmotionEntailmentPostprocessor,
)
with open(f'emotion_classifier.pkl','rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')

# initializing global variables
global config 
global tokenizer
global model2
global preprocessor 
global postprocessor
config = RecconEmotionEntailmentConfig.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/reccon_emotion_entailment/config.json"
)
tokenizer = RecconEmotionEntailmentTokenizer.from_pretrained("roberta-base")
model2 = RecconEmotionEntailmentModel.from_pretrained(
    "https://storage.googleapis.com/sgnlp/models/reccon_emotion_entailment/pytorch_model.bin",
    config=config,
)
preprocessor = RecconEmotionEntailmentPreprocessor(tokenizer)
postprocessor = RecconEmotionEntailmentPostprocessor()


def getInputBatch(emotion,p1,p2,p3,p4,p5):
    input_batch = {"emotion": [emotion, emotion, emotion, emotion, emotion],
        "target_utterance": [
            p5,
            p5,
            p5,
            p5,
            p5
        ],
        "evidence_utterance": [
            p1,
            p2,
            p3,
            p4,
            p5

        ],
        "conversation_history": [
            p1 +" " + p2 + " " + p3 + " "+ p4 + " " +p5,
            p1 +" " + p2 + " " + p3 + " "+ p4 + " " +p5,
            p1 +" " + p2 + " " + p3 + " "+ p4 + " " +p5,
            p1 +" " + p2 + " " + p3 + " "+ p4 + " " +p5,
            p1 +" " + p2 + " " + p3 + " "+ p4 + " " +p5
        ],
    }
    return input_batch


@app.route('/', methods=['GET', 'POST'])
def main():
    
    if flask.request.method == 'GET':
        return(flask.render_template('model2.html'))
    
    if flask.request.method == 'POST':
        p1 = flask.request.form['p1']
        p2 = flask.request.form['p2']
        p3 = flask.request.form['p3']
        p4 = flask.request.form['p4']
        p5 = flask.request.form['p5']
       
        prediction1 = model.predict([p1])[0]
        prediction2 = model.predict([p2])[0]
        prediction3 = model.predict([p3])[0]
        prediction4 = model.predict([p4])[0]
        prediction5 = model.predict([p5])[0]

        inputText = [p1,p2,p3,p4,p5]
        theList = [prediction1,prediction2, prediction3, prediction4, prediction5]


        list2 = []
        for emo in theList:
            if(emo != 'neutral'):
                list2.append(emo)
        

        emotion = ""
        if len(list2) == 0:
            emotion = 'neutral'
        else:
            emotion = mode(list2)

        score = round(theList.count(emotion) / 5, 3)

        input_batch = getInputBatch(emotion, p1,p2,p3,p4,p5)
        tensor_dict = preprocessor(input_batch)
        raw_output = model2(**tensor_dict)
        output = postprocessor(raw_output)

        evidence = []
        for i,val in enumerate(output):
            if val==1:
                evidence.append(inputText[i])
        
        return flask.render_template('model2.html',
                                     result=emotion,
                                     score=score,
                                     output=evidence
                                     )
if __name__ == '__main__':
    app.run(port=8090,debug=True)
