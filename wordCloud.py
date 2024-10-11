import nltk
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt



input_folder_path = os.path.join(".", "preprocessed_files")
output_folder_path = os.path.join(".", "wordClouds")

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

for file in os.listdir(input_folder_path):
    file_path = os.path.join(input_folder_path, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
    output_file_path = os.path.join(output_folder_path, f"wordCloud_{file}.png")
    wordcloud.to_file(output_file_path)
    print(f"Word clouds generated and saved! {output_file_path}")



