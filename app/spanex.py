from sgnlp.models.span_extraction import (
    RecconSpanExtractionConfig,
    RecconSpanExtractionModel,
    RecconSpanExtractionTokenizer,
    RecconSpanExtractionPreprocessor,
    RecconSpanExtractionPostprocessor,
)

# Load model
config = RecconSpanExtractionConfig.from_pretrained("config.json")
tokenizer = RecconSpanExtractionTokenizer.from_pretrained(
    "mrm8488/spanbert-finetuned-squadv2"
)
model = RecconSpanExtractionModel.from_pretrained(
    "spanex.bin",
    config=config,
)
preprocessor = RecconSpanExtractionPreprocessor(tokenizer)
postprocessor = RecconSpanExtractionPostprocessor()


def add_period(conversation):
    for i in range(len(conversation)):
        if conversation[i][-1].isalnum():
            conversation[i] += '.'

    return conversation


def predict(emotion, target, evidence, history):
    tries = len(evidence)
    emotion = [emotion] * tries
    target = [target] * tries
    history = [history] * tries

    input_batch = {"emotion": emotion, "target_utterance": target, "evidence_utterance": evidence,
                   "conversation_history": history}

    tensor_dict, evidences, examples, features = preprocessor(input_batch)
    raw_output = model(**tensor_dict)
    context, evidence_span, probability = postprocessor(
        raw_output, evidences, examples, features)

    context_cat = []
    for part in context:
        context_cat += part

    evidence_cat = []
    for part in evidence_span:
        evidence_cat += part

    probability_cat = []
    for part in probability:
        probability_cat += part

    res = []

    for i in range(len(context_cat)):
        if evidence_cat[i] == 1:
            res.append([i, probability_cat[i]])

    return res


def run(emotion, conversation):
    evidence_pool = add_period(conversation)
    res = [[]] * len(conversation)
    for i in range(len(emotion)):
        if emotion[i] != "non-negative":
            emo = emotion[i]
            target = conversation[i]
            evidence = evidence_pool[0:i]
            history = ' '.join(evidence)
            curr_res = predict(emo, target, evidence, history)
            res[i] = res[i] + curr_res

    return res
