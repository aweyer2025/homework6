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

        # Process each line to remove stop words, keeping original tab spacing and structure
        for line in lines:
            # Split line into sections using tabs
            sections = line.split("\t")

            # Process only the sections containing text (review title and body)
            for i in range(len(sections)):
                if i > 0:  # Skip first section if it is a date
                    words = word_tokenize(sections[i])
                    filtered_words = [word for word in words if word.lower() not in stop_words]
                    sections[i] = " ".join(filtered_words)  # Rebuild the section with filtered words

            # Reconstruct the line with preserved tab spacing
            filtered_line = "\t".join(sections)
            all_filtered_text.append(filtered_line.strip() + "\n")  # Retain newline at the end of each line

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



  

        # Extract lemmatized words while keeping original formatting and adding spaces as needed
        lemmatized_text = []
        previous_token_was_space = False

        for token in doc:
            if token.is_space:
                lemmatized_text.append(token.text)  # Preserve spaces
                previous_token_was_space = True
            else:
                if previous_token_was_space or len(lemmatized_text) == 0:
                    lemmatized_text.append(token.lemma_)
    

        # Join the lemmatized tokens while keeping the original spaces
        lemmatized_output_text = " ".join(lemmatized_text)




        # Define new output path for lemmatized stop word filtered text
        lemmatized_output_file_name = f"{os.path.splitext(file)[0]}_lemmatized_filtered.txt"
        lemmatized_output_file_path = os.path.join(output_folder_path, lemmatized_output_file_name)

        # Write filtered and lemmatized text to a new file
        with open(lemmatized_output_file_path, 'w', encoding='utf-8') as lemmatized_output_file:
            lemmatized_output_file.write(lemmatized_output_text)

        print(f"Lemmatized and filtered text has been saved to {lemmatized_output_file_path}")

    except Exception as e:
        print(f"Could not process file {file}: {e}")
