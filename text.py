'''
from textblob import TextBlob

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment  # Returns polarity and subjectivity


#character_sentiments = {character: analyze_sentiment(dialogue) for character, dialogue in dataset.items()}
character="a1"
dialogue="I will avenge my family, no matter the cost"
character_sentiments = {character: analyze_sentiment(dialogue)}
print(character_sentiments)
'''

# VADER
'''
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

dialogue = "I will avenge my family, no matter the cost"
sentiment = sia.polarity_scores(dialogue)
print(sentiment)  # Output: {'neg': 0.0, 'neu': 0.4, 'pos': 0.6, 'compound': 0.4215}
'''

'''
# Hugging Face
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")
dialogue = "I will avenge my family, no matter the cost"
sentiment = sentiment_pipeline(dialogue)
print(sentiment)  # Output: [{'label': 'POSITIVE', 'score': 0.99}]
'''

import kagglehub

# Download latest version
path = kagglehub.dataset_download("saurabhbadole/game-of-thrones-book-dataset")

print("Path to dataset files:", path)