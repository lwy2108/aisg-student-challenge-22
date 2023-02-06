import emoclass
import spanex

emotion, conversation = emoclass.classify(
    [
        "hi how are you",
        "you ugly piece of shit",
        "you are so beautiful",
        "good day",
        "fuck you",
        "i hate you",
        "i love you",
        "the weather is fine"
    ]
)

print(spanex.run(emotion, conversation))
