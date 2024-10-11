import spacy
import os
import chardet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.data.path.append('/home/aweyer/nltk_data')

nltk_data_dir = os.path.join(".", "nltk_data")
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)

input_folder_path = os.path.join(".", "input")
output_folder_path = os.path.join(".", "preprocessed_files")
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

for file in os.listdir(input_folder_path):
    file_path = os.path.join(input_folder_path, file)
    all_filtered_text = []  
    try:
        with open(file_path, 'rb') as binary_file:
            raw_data = binary_file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
        with open(file_path, 'r', encoding=encoding, errors='ignore') as text_file:
            lines = text_file.readlines()  
        for line in lines:
            sections = line.split("\t")
            for i in range(len(sections)):
                if i > 0:  
                    words = word_tokenize(sections[i])
                    filtered_words = [word for word in words if word.lower() not in stop_words]
                    sections[i] = " ".join(filtered_words)  

            filtered_line = "\t".join(sections)
            all_filtered_text.append(filtered_line.strip() + "\n")  

        stop_words_output_file_name = f"{os.path.splitext(file)[0]}_removedStopWords.txt"
        stop_word_output_file_path = os.path.join(output_folder_path, stop_words_output_file_name)

        with open(stop_word_output_file_path, 'w', encoding='utf-8') as stop_words_output_file:
            stop_words_output_file.writelines(all_filtered_text)  
        print(f"Updated file with removed stop words is saved to {stop_word_output_file_path}")

        with open(stop_word_output_file_path, 'r', encoding='utf-8', errors='ignore') as stop_lemmatize_file:
            stop_text = stop_lemmatize_file.read()

        doc = nlp(stop_text)

        lemmatized_text = []
        previous_token_was_space = False

        for token in doc:
            if token.is_space:
                lemmatized_text.append(token.text) 
                previous_token_was_space = True
            else:
                if previous_token_was_space or len(lemmatized_text) == 0:
                    lemmatized_text.append(token.lemma_)
        lemmatized_output_text = " ".join(lemmatized_text)
        lemmatized_output_file_name = f"{os.path.splitext(file)[0]}_lemmatized_filtered.txt"
        lemmatized_output_file_path = os.path.join(output_folder_path, lemmatized_output_file_name)
        with open(lemmatized_output_file_path, 'w', encoding='utf-8') as lemmatized_output_file:
            lemmatized_output_file.write(lemmatized_output_text)

        print(f"Lemmatized and filtered text has been saved to {lemmatized_output_file_path}")
        os.remove(stop_word_output_file_path)
        print(f"Deleted stop words file: {stop_word_output_file_path}")

    except Exception as e:
        print(f"Could not process file {file}: {e}")
