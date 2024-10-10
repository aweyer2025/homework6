import nltk
import spacy
import os
import chardet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Append NLTK data path if necessary
nltk.data.path.append('/home/aweyer/nltk_data')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Define input and output paths
input_folder_path = os.path.join(".", "input")
output_folder_path = os.path.join(".", "preprocessed_files")

#load spacy model
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

        #Writing stop words to stop word output file
        with open(stop_word_output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.writelines(all_filtered_text)  # Use writelines to maintain formatting
        print(f"Updated file with removed stop words is saved to {stop_word_output_file_path}")

        #opening file for lemmatization 
        with open(file_path, 'r', encoding=encoding) as file:
            text = file.read()
        # spacy processing
        doc = nlp(text)

        # Extract lemmatized words
        lemmatized_text = " ".join([token.lemma_ for token in doc])

        #Writing lemmatized text to lemmitezed output file
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(lemmatized_text)

        print(f"Lemmatized text has been saved to {output_file}")

        

        

    except Exception as e:
        print(f"Could not process file {file}: {e}")

