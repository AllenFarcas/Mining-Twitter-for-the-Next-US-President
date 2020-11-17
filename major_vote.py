from collections import Counter
import csv

# Writer information
csv_voted  = open("vote_all.csv", "w")
fieldnames = ['Tweet ID', 'text', 'SentiStrength', 'Vader', 'Textblob', 'Vote']
writer     = csv.DictWriter(csv_voted, fieldnames=fieldnames)
writer.writeheader()

# Reader
csv_SentiStrength = open("SentiStrength_labels.csv", "r")
csv_Vader         = open("Tweet_sentiment_Vader.csv", "r")
csv_Textblob      = open("Textblob_sentiment.csv", "r")

csv_Textblob.readline()

readCSV1 = csv.reader(csv_SentiStrength, delimiter=',')
readCSV2 = csv.reader(csv_Vader, delimiter=',')
readCSV3 = csv.reader(csv_Textblob, delimiter=',')

results = []
for row1, row2, row3 in zip(readCSV1, readCSV2, readCSV3):
    if len(row2) < 3:
        continue
    # print(row2)
    ID_1 = int(row1[0])
    ID_2 = int(row2[0].replace("\t", "").replace("\ufeff",""))
    ID_3 = int(row3[0])
    assert ID_1==ID_2 and ID_2==ID_3 and ID_1==ID_3
    Result_1 = row1[2].replace("\t", "")
    Result_2 = row2[2].replace("\t", "")
    Result_3 = row3[2].replace("\t", "")
    x = Counter([Result_1, Result_2, Result_3])
    results.append(x)
    if x.most_common(1)[0][1] == 1:
        vote = 'Neutral'
    else:
        vote = x.most_common(1)[0][0]
    writer.writerow({'Tweet ID': ID_1, 'text': row1[1], 'SentiStrength': Result_1, 'Vader': Result_2, 'Textblob': Result_3, 'Vote': vote})

csv_voted.close()
csv_SentiStrength.close()
csv_Vader.close()
csv_Textblob.close()

result_111 = sum(1 for r in results if r.most_common(1)[0][1]==1)
result_2__ = sum(1 for r in results if r.most_common(1)[0][0]=='Positive' and r.most_common(1)[0][1]==2)
result__2_ = sum(1 for r in results if r.most_common(1)[0][0]=='Neutral'  and r.most_common(1)[0][1]==2)
result___2 = sum(1 for r in results if r.most_common(1)[0][0]=='Negative' and r.most_common(1)[0][1]==2)
result_300 = sum(1 for r in results if r.most_common(1)[0][0]=='Positive' and r.most_common(1)[0][1]==3)
result_030 = sum(1 for r in results if r.most_common(1)[0][0]=='Neutral'  and r.most_common(1)[0][1]==3)
result_003 = sum(1 for r in results if r.most_common(1)[0][0]=='Negative' and r.most_common(1)[0][1]==3)
print(f"(Pos,Neu,Neg)=(1,1,1): {result_111}; {100*result_111/len(results):0.2f}%")
print(f"(Pos,Neu,Neg)=(2,-,-): {result_2__}; {100*result_2__/len(results):0.2f}%")
print(f"(Pos,Neu,Neg)=(-,2,-): {result__2_}; {100*result__2_/len(results):0.2f}%")
print(f"(Pos,Neu,Neg)=(-,-,2): {result___2}; {100*result___2/len(results):0.2f}%")
print(f"(Pos,Neu,Neg)=(3,0,0): {result_300}; {100*result_300/len(results):0.2f}%")
print(f"(Pos,Neu,Neg)=(0,3,0): {result_030}; {100*result_030/len(results):0.2f}%")
print(f"(Pos,Neu,Neg)=(0,0,3): {result_003}; {100*result_003/len(results):0.2f}%")
print(f"Total: {len(results)}")
assert len(results) == (result_111+result_2__+result__2_+result___2+result_300+result_030+result_003)