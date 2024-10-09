import spacy
import chardet

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Define file paths
input_file = "/home/aweyer/my_windows_folder/SQL_Projects/Homework6/input/uk_england_london_britannia_international_hotel"   # Replace with your input file path
output_file = "/home/aweyer/my_windows_folder/SQL_Projects/Homework6/output/output.txt"  # Replace with your desired output file path


import chardet

# Detect the file's encoding
with open(input_file, 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    print(result)

# The detected encoding can be used to read the file
encoding = result['encoding']

# Read the file with the detected encoding
with open(input_file, 'r', encoding=encoding) as file:
    text = file.read()


# Process the text using spaCy
doc = nlp(text)

# Extract lemmatized words
lemmatized_text = " ".join([token.lemma_ for token in doc])

# Write the lemmatized text to the output file
with open(output_file, 'w', encoding='utf-8') as file:
    file.write(lemmatized_text)

print(f"Lemmatized text has been saved to {output_file}")
