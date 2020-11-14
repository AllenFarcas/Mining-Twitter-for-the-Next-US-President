# SentiStrength.jar is commercial product (it can be obtained for free with academic use, but I think using free version should be enough)
# Using Windows IDE - have to extract texts as file (with new lines)

import pandas as pd
import re

filename = "../dataset_filt.jsonl"
outputname = "SentiStrength.txt"
chunksize = 100

#Data preprocessing to removes @usernames,urls,symbols and makes all text lowercase
def preprocess_text(text):
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL', text)
    text = re.sub('@[^\s]+','', text)
    text = text.lower().replace("ё", "е")
    text = re.sub('[^a-zA-Zа-яА-Я1-9]+',' ', text)
    text = re.sub(' +',' ', text)
    if text.strip() == "":
        text = "empty"
    return text.strip()

chunks = pd.read_json(filename, orient="records", chunksize=chunksize, lines=True, nrows=5000000)
with open(outputname, 'w', encoding="utf-8") as writer:
    writer.write("\n") #Write new line since SentiStrength Windows IDE reads first line as header
    i = 1
    j = 0
    for chunk in chunks:
        if i % 1000 == 0:
            print("Chunk " + str(i))
        text = [preprocess_text(t) for t in chunk.iloc[:]['full_text']]
        if len(text) != 100:
            print(len(text))
        writer.write("\n".join(text))
        i = i + 1
        j += len(text)
    print(i, j)