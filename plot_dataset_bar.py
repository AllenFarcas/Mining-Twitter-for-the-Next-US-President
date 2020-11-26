import csv
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# plt.style.use('ggplot')


bar_dict = {'Democratic':{'Positive':0, 'Negative':0, 'Neutral':0},\
            'Republican':{'Positive':0, 'Negative':0, 'Neutral':0},}
# Reader
with open('filtered.csv', newline='') as csvfile:
    csvfile.readline()
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        party = row[-1]
        senti = row[-2]
        if party == 'Democratic' or party == 'Republican':
            bar_dict[party][senti] += 1

csvfile.close()

total = sum([sum(dd.values()) for dd in bar_dict.values()])
total_dem = sum(bar_dict['Democratic'].values())
total_rep = sum(bar_dict['Republican'].values())
print(bar_dict,total)

# Plot
width = 0.32
space = 0.04
ind = np.arange(3)

plt.figure()
plt.bar(ind            , 100*np.array(list(bar_dict['Democratic'].values()))/total_dem, width, color='#019bd8', label='Democratic')
plt.bar(ind+width+space, 100*np.array(list(bar_dict['Republican'].values()))/total_rep, width, color='#D81C28', label='Republican')

plt.ylabel('Percentage (%)')
# plt.title('Scores by group and gender')

plt.xticks(ind + width/2 + space/2, ('Positive', 'Negative', 'Neutral'))
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('myimage.png', format='png', dpi=1200)

plt.show()

# result_111 = sum(1 for r in results if r.most_common(1)[0][1]==1)
# result_2__ = sum(1 for r in results if r.most_common(1)[0][0]=='Positive' and r.most_common(1)[0][1]==2)
# result__2_ = sum(1 for r in results if r.most_common(1)[0][0]=='Neutral'  and r.most_common(1)[0][1]==2)
# result___2 = sum(1 for r in results if r.most_common(1)[0][0]=='Negative' and r.most_common(1)[0][1]==2)
# result_300 = sum(1 for r in results if r.most_common(1)[0][0]=='Positive' and r.most_common(1)[0][1]==3)
# result_030 = sum(1 for r in results if r.most_common(1)[0][0]=='Neutral'  and r.most_common(1)[0][1]==3)
# result_003 = sum(1 for r in results if r.most_common(1)[0][0]=='Negative' and r.most_common(1)[0][1]==3)
# print(f"(Pos,Neu,Neg)=(1,1,1): {result_111}; {100*result_111/len(results):0.2f}%")
# print(f"(Pos,Neu,Neg)=(2,-,-): {result_2__}; {100*result_2__/len(results):0.2f}%")
# print(f"(Pos,Neu,Neg)=(-,2,-): {result__2_}; {100*result__2_/len(results):0.2f}%")
# print(f"(Pos,Neu,Neg)=(-,-,2): {result___2}; {100*result___2/len(results):0.2f}%")
# print(f"(Pos,Neu,Neg)=(3,0,0): {result_300}; {100*result_300/len(results):0.2f}%")
# print(f"(Pos,Neu,Neg)=(0,3,0): {result_030}; {100*result_030/len(results):0.2f}%")
# print(f"(Pos,Neu,Neg)=(0,0,3): {result_003}; {100*result_003/len(results):0.2f}%")
# print(f"Total: {len(results)}")
# assert len(results) == (result_111+result_2__+result__2_+result___2+result_300+result_030+result_003)