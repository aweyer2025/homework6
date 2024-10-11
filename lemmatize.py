import spacy
import os
import chardet

nlp = spacy.load("en_core_web_sm")
input_folder_path = os.path.join(".", "input")
output_folder_path = os.path.join(".", "preprocessed_files")

input_file = os.path.join(input_folder_path, "uk_england_london_britannia_international_hotel")  # Specify the actual file name
output_file = os.path.join(output_folder_path, "lemmatized_output.txt")

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Detect the file's encoding
with open(input_file, 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    print(result)
encoding = result['encoding']
with open(input_file, 'r', encoding=encoding) as file:
    text = file.read()
doc = nlp(text)

lemmatized_text = " ".join([token.lemma_ for token in doc])

with open(output_file, 'w', encoding='utf-8') as file:
    file.write(lemmatized_text)

print(f"Lemmatized text has been saved to {output_file}")
