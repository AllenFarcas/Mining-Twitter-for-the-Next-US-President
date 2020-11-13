# SentiStrength.jar is commercial product (it can be obtained for free with academic use, but I think using free version should be enough)
# Using Windows IDE - have to extract texts as file (with new lines)

import pandas as pd
import re

filename = "../Datasets/dataset_filt_small.jsonl"
outputname = "SentiStrength_test.txt"
chunksize = 100

#Data preprocessing to removes @usernames,urls,symbols and makes all text lowercase
def preprocess_text(text):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', text)
    text = re.sub('@[^\s]+','', text)
    text = text.lower().replace("ё", "е")
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+',' ', text)
    text = re.sub(' +',' ', text)
    return text.strip()

chunks = pd.read_json(filename, orient="records", chunksize=chunksize, lines=True, nrows=5000000)
with open(outputname, 'w') as writer:
    writer.write("\n") #Write new line since SentiStrength Windows IDE reads first line as header
    for chunk in chunks:
        text = [preprocess_text(t) for t in chunk.iloc[:]['full_text']]
        writer.write("\n".join(text))