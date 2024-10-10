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

# Function to analyze sentiment for a given text
def analyze_sentiment(text):
    # Get sentiment scores
    scores = sia.polarity_scores(text)
    return scores

# Loop through each file in the input folder
for file in os.listdir(input_folder_path):
    file_path = os.path.join(input_folder_path, file)

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split text into paragraphs
    paragraphs = text.split('\t')  

    # Analyze sentiment for each paragraph
    paragraph_scores = []
    for i, paragraph in enumerate(paragraphs):
        if paragraph.strip():  # Skip empty paragraphs
            scores = analyze_sentiment(paragraph)
            paragraph_scores.append((i, scores))
            # print(f"Paragraph {i + 1} Scores: {scores}")


    sentances = text.split('.')

    sentance_scores=[]

    for i, sentance in enumerate(sentances):
        if sentance.strip():
            scores = analyze_sentiment(sentance)
            sentance_scores.append((i,scores))
            # print(f"Sentance {i + 1} Scores: {scores}")

    document = text
    documents=" "
    documentScore = analyze_sentiment(document)
    print (f"score for document as a whole {documentScore}")
  





    # Output sentiment scores for each paragraph
    output_file_path = os.path.join(output_folder_path, f"analysed_{file}\n")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(f"Whole document sentiment score {documentScore}")
        for para_index, scores in paragraph_scores:
            output_file.write(f"Paragraph {para_index + 1} Scores: {scores}\n")
        for sentance_index, scores in sentance_scores:
            output_file.write(f"Sentence {sentance_index} Scores: {scores}\n")
        

    print(f"Sentiment analysis completed for file: {file}\n")
