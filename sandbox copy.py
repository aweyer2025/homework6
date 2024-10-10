import nltk
import spacy
import os
import chardet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Append NLTK data path if necessary
nltk.data.path.append('/home/aweyer/nltk_data')

# Define the relative path for the nltk_data folder
nltk_data_dir = os.path.join(".", "nltk_data")

# Download the nltk packages to the relative folder
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)

# Define input and output paths
input_folder_path = os.path.join(".", "input")
output_folder_path = os.path.join(".", "preprocessed_files")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize stop words set
stop_words = set(stopwords.words('english'))

# Process each file in the input folder
for file in os.listdir(input_folder_path):
    file_path = os.path.join(input_folder_path, file)

    all_filtered_text = []  # To store each line after removing stop words

    try:
        # Detect the encoding of the file
        with open(file_path, 'rb') as binary_file:
            raw_data = binary_file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        # Opening file for stop word removal
        with open(file_path, 'r', encoding=encoding, errors='ignore') as text_file:
            lines = text_file.readlines()  # Read the file line-by-line to maintain the original format

        # Process each line to remove stop words
        for line in lines:
            words = word_tokenize(line)
            filtered_words = [word for word in words if word.lower() not in stop_words]
            # Join the filtered words back together with spaces and retain the original newline character
            filtered_line = " ".join(filtered_words)
            all_filtered_text.append(filtered_line + "\n")

        # Write the filtered text to a new file (stop words removed)
        stop_words_output_file_name = f"{os.path.splitext(file)[0]}_removedStopWords.txt"
        stop_word_output_file_path = os.path.join(output_folder_path, stop_words_output_file_name)

        with open(stop_word_output_file_path, 'w', encoding='utf-8') as stop_words_output_file:
            stop_words_output_file.writelines(all_filtered_text)  # Use writelines to maintain formatting
        print(f"Updated file with removed stop words is saved to {stop_word_output_file_path}")

        # Opening stop words-removed file for lemmatization
        with open(stop_word_output_file_path, 'r', encoding='utf-8', errors='ignore') as stop_lemmatize_file:
            stop_text = stop_lemmatize_file.read()

        # spaCy processing
        doc = nlp(stop_text)

        # Extract lemmatized words
        lemmatized_text = " ".join([token.lemma_ for token in doc])

        # Define new output path for lemmatized stop word filtered text
        lemmatized_output_file_name = f"{os.path.splitext(file)[0]}_lemmatized_filtered.txt"
        lemmatized_output_file_path = os.path.join(output_folder_path, lemmatized_output_file_name)

        # Write filtered and lemmatized text to a new file
        with open(lemmatized_output_file_path, 'w', encoding='utf-8') as lemmatized_output_file:
            lemmatized_output_file.write(lemmatized_text)

        print(f"Lemmatized and filtered text has been saved to {lemmatized_output_file_path}")

    except Exception as e:
        print(f"Could not process file {file}: {e}")
