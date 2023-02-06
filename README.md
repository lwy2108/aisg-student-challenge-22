# aisg-student-challenge-22

Welcome to "Casual"! This app predicts negative emotions 
in your conversations(if any) and further analyses the causes for it.

Follow the steps below to get an analysis for your conversations:
1) Extract a conversation from WhatsApp (or format each line as Sender: message)
2) Paste the conversation into Causal
3) Hit run! "Casual" app will analyze the chat, and output the predictions

Note: you can hover over individual messages to see their causation visualized.

Causal was written with a Flask backend. To run the server, follow this tutorial: https://flask.palletsprojects.com/en/2.2.x/quickstart/

## required packages
1. sgnlp
2. scikit-learn
3. flask

# test.py
Test the model with your own inputs by running them through this file

# emoclass.ipynb
You can look in the notebook to see how we arrived at our emoclass model.
