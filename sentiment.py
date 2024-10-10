import nltk
import os
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon if you haven't done so
nltk.download('vader_lexicon')

# Initialize the sentiment intensity analyzer
sia = SentimentIntensityAnalyzer()

# Define input and output paths
input_folder_path = os.path.join(".", "preprocessed_files")
output_folder_path = os.path.join(".", "analysed_files")


for file in os.listdir(input_folder_path):
    file_path=os.path.join(input_folder_path, file)

    def analyze_sentiment(text):
        # Get sentiment scores
        scores = sia.polarity_scores(text)
        return scores

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    sentiment_scores = analyze_sentiment(text)


print(f"Sentiment Scores for document as a whole {sentiment_scores}\n")
