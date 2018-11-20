from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob

train = [
    ('I love this sandwich.', 'pos'),
    ('This is an amazing place!', 'pos'),
    ('I feel very good about these beers.', 'pos'),
    ('This is my best work.', 'pos'),
    ("What an awesome view", 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
    ('He is my sworn enemy!', 'neg'),
    ('My boss is horrible.', 'neg')
]

classificator = NaiveBayesClassifier(train)

textBlobParragraph = TextBlob("The beer was amazing. But the hangover was horrible. "
                "My boss was not pleased. "
                "Leonardo Bouchan is a good Developer.", classifier=classificator)

# Complete Classification of the TextBlob
    # print(textBlobParragraph)
    # print(textBlobParragraph.classify())

for sentence in textBlobParragraph.sentences:
    print("Sentence : " , sentence)
    print("  Classify : " , sentence.classify())
    print("  Detected Language : " , sentence.detect_language())
    print("  Tags : " , sentence.tags) # Reference of POS https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    print("  Noun Phrases : " , sentence.noun_phrases)
    print("  Word List : " , sentence.words)
    print("  Sentiment : " , sentence.sentiment)
    print("  Title : " , sentence.title()) # Is like to create a Title
    print("                                   ")