import nltk
import os
import chardet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.data.path.append('/home/aweyer/nltk_data')
nltk.download('stopwords')
nltk.download('punkt')
input_folder_path = os.path.join(".", "input")
output_folder_path = os.path.join(".", "preprocessed_files")

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

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
                words = word_tokenize(sections[i])  
                filtered_words = [word for word in words if word.lower() not in stop_words] 
                sections[i] = " ".join(filtered_words)
            filtered_line = "\t".join(sections)  
            all_filtered_text.append(filtered_line + "\n")  

        stop_words_output_file_name = f"{os.path.splitext(file)[0]}_removedStopWords.txt"
        stop_word_output_file_path = os.path.join(output_folder_path, stop_words_output_file_name)

        with open(stop_word_output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.writelines(all_filtered_text)  

        print(f"Updated file with removed stop words saved to {stop_word_output_file_path}")

    except Exception as e:
        print(f"Could not process file {file}: {e}")
