# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 09:57:07 2019

@author: User
"""


import xlrd
from fgutil import *

print('Start Feature Extraction')        
cnBigramJiebaPos = {}

f = filterWeiBo()
n = Ngram()

workbook = xlrd.open_workbook("data/weibo_everything2.xlsx")
worksheet = workbook.sheet_by_index(4)
nrows = worksheet.nrows
tenP = int(nrows / 10)
post_d = {}
emop_d = {}
for i in range(1, nrows):
    value = worksheet.cell_value(i, 3)
    value = f.filterCell(value)
    cn2gramJiebaPos = n.cnNgramPos(value)
    cn2gramJiebaPos, list_of_words = n.stopwRemove(cn2gramJiebaPos)
    emo_label=[0,0,0,0,0]
    for j in range(4, 9):
        if worksheet.cell_value(i, j) == 'Y':
            emo_label[j-4] = 1
            for word in list_of_words:
                if word in emop_d:
                    emop_d[word][j-4] = emop_d[word][j-4]+1
                else:
                    temp_l = [0,0,0,0,0]
                    temp_l[j-4] = 1
                    emop_d[word] = temp_l[:]
    post = Post(i,value,cn2gramJiebaPos,emo_label)
    post_d.update(post.get_post())
    addWordsPos(cnBigramJiebaPos, cn2gramJiebaPos)
    if (i % tenP == 0):
        print('{0}%'.format(int(i/nrows * 100)), end = ' ')
    # print(cn2gramJiebaPos)

with open("data/data_file.json",'w',encoding='utf-8') as write_file:
    json.dump(post_d, write_file, ensure_ascii=False)

for word, value in emop_d.items():
    norm = sum(value)
    value = [vl / norm for vl in value]
    emop_d[word] = value

with open("data/emop_file.json",'w',encoding='utf-8') as write_file:
    json.dump(emop_d, write_file, ensure_ascii=False)
    
with open("data/WeiBoCorpusBigramJiebaEngPos.txt",'w',encoding='utf-8') as outFile:
    for key, [x, y, z] in cnBigramJiebaPos.items():
        outFile.write(f'{key} {x} {y} {z}\n')

print('100%\nComplete Feature Extraction')

print('Start TFIDF')

gram_count = 0
term_dict = {}
ti = TFIDF(nrows - 1) 
infile = open("data/WeiBoCorpusBigramJiebaEngPos.txt",'r',encoding='utf-8').read().split('\n')
outfile = 'outcome/testTFIDFBigramJiebaEngPos.txt'
leng = len(infile) - 1
count = 0

f_l, df_l = [], []
for terms in infile:
    try:
        term, pos, f, df = terms.split(' ',3)
        term_dict[term] = [pos]
        f_l.append(f)
        df_l.append(df)
        count += 1
    except:
        if count == leng:
            print('completed')
        else:
            print('not complete')  

ti_l = ti.getTFIDF(f_l, df_l) 

i = 0
for z in term_dict.items():
    x, y = z
    y += ti_l[i]
    i += 1
            
with open(outfile,'w',encoding='utf-8') as outFile:
    for key, [u, v, w, x, y, z] in sorted(term_dict.items(), key = lambda item : item[1][5], reverse = True):
        outFile.write(f'{key} {u} {v} {w} {x} {y} {z}\n')

sort_term_dict = dict(sorted(term_dict.items(), key = lambda item : item[1][5], reverse = True))
#n.stopwRemove(sort_term_dict)
with open("data/data_file2.json",'w',encoding='utf-8') as write_file:
    json.dump(sort_term_dict, write_file, ensure_ascii=False)
        
n.stopwRemove(outfile)
n.stopwRemove("data/data_file2.json")