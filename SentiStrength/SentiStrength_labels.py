import pandas as pd
import numpy as np
import re

filename = "SentiStrength+results.txt"
jsonname = "../dataset_filt.jsonl"
outputname = "SentiStrength_labels.csv"
chunksize = 100

chunks = pd.read_csv(filename, sep="\t", chunksize=chunksize)
datachunks = pd.read_json(jsonname, orient="records", chunksize=chunksize, lines=True, nrows=5000000)

i = 1
for data in datachunks:
    chunk = chunks.get_chunk()
    sentimentValue = chunk['Positive'] + chunk['Negative']
    sentimentString = np.select([sentimentValue.ge(2), sentimentValue.le(-2), sentimentValue.lt(2) & sentimentValue.gt(-2)],['Positive', 'Negative', 'Neutral'])
    #sentimentCsv = pd.DataFrame({'ID': data['id_str'], 'Value': sentimentValue, 'String': sentimentString, 'Text': chunk['Translation']})
    sentimentCsv = pd.DataFrame({'ID': data['id_str'], 'Text': data['full_text'], 'Sentiment': sentimentString})
    sentimentCsv.to_csv(outputname, index=False, mode='a', header=False)
    i = i + 1
    if i % 1000 == 0:
        print('Chunk ' + str(i))