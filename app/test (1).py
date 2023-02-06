import emoclass
import spanex

"""
Jeremy: how was dinner at the restaurant
Ron: the waiter at our table was very rude
Jeremy: what did he do
Ron: he was yelling and cussing at me
Jeremy: oh my god what a rude guy
Ron: i know right i am never going here again
"""

test_list:list = [
    [
        "how was the dinner at the restaurant",
        "the waiter at our table was very rude", # cause for anger
        "what did he do", 
        "he was yelling and cussing at me", # cause for anger 
        "oh my god what a rude guy",
        "i know right i am never going there again"
    ],
    # [[], [], [], [], [[1, 0.8275642271524505], [3, 0.8982184494584358]], []]
    # ['non-negative', 'anger', 'non-negative', 'non-negative', 'anger', 'non-negative']
    [
        "how was the exam today",
        "it was so horrible", # cause for sadness
        "why what happened", 
        "the questions were so hard i felt so stressed", 
        "please dont feel stressed be happy that its now over",
        "thanks"
    ],
    # [[], [], [], [[1, 0.8443898846902494]], [[1, 0.8000024505951602]], []]
    # ['non-negative', 'sadness', 'non-negative', 'sadness', 'sadness', 'non-negative']
    [
        "why did you divorce with him",
        "he neglected my child abused me and even called me ugly", # cause for sadness 
        "thats horrible im so sorry you have to go through that",
        "its okay everything is good now",
        "i hope you find a better man in the future"
    ],
    # [[], [], [[1, 0.916718050370189]], [], []]
    # ['non-negative', 'sadness', 'sadness', 'non-negative', 'non-negative']
    [
        "how is your day today",
        "my manager was pissed at me and called me an asshole in front of everyone", # cause for emotion x2
        "that is so rude of him and must be so embarrassing for you",
        "i know",
        "what are you gonna do",
        "i was so embarrassed and humiliated by him so i submitted my resignation letter and left"
    ]
    # [[], [], [[1, 0.79726098476648]], [], [], [[1, 0.7764969350922738]]]
    # ['non-negative', 'anger', 'anger', 'non-negative', 'non-negative', 'sadness']
]

for i in test_list:
    emotion, conversation = emoclass.classify(    
        i
    )

    print(spanex.run(emotion, conversation))
    print(emotion)
