
#import json
#
#stopw_list = []
#clean = []
#stopw = []
#
#cn, en = not True, True
##filename = 'dict/stopwords_all.json'
##filename = 'dict/stopwords_zh.json'
#filename = 'dict/stopwords_en.json'
#
#if cn :
#    stopw_list += open('dict/stopwords_zh.txt',encoding='utf-8').read().split('\n')
#if en :
#    stopw_list += open('dict/stopwords_en_nltk.txt').read().split('\n')
#
#
#with open(filename,'w',encoding='utf-8') as outf:
#    json.dump(stopw_list, outf, ensure_ascii=False)
#
#filename = 'dict/stopwords_all.json'
#stopw_all = json.load(open(filename,'r',encoding='utf-8'))
#print(stopw_all)
#
#if cn and not en:
#    print('1')
#elif not cn and en:
#    print('2')
#else:
#    print('3')    

import numpy as np
import re

# print(np.array([1]))

import csv
import json
from jieba import posseg

sen = "玩了一下午轮滑！so happy 好高兴"
new_sen = list(filter(None, re.split("\s+|[^\u4e00-\u9FFFA-Za-z']+|[！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.]+",sen)))

stopw_all = open('dict/stopwords_all.json','r',encoding='utf-8')
stop_d = json.load(stopw_all)

fvs = [word for sen in new_sen for word, flag in posseg.cut(sen) if word not in stop_d]

print(fvs)

data_dict = open('data/data_file2.json','r',encoding='utf-8')
emop_dict = open('data/emop_file.json','r',encoding='utf-8')
data_d = json.load(data_dict)
emop_d = json.load(emop_dict)

for fv in fvs:
    try:
        print(fv, data_d.get(fv), emop_d.get(fv))
    except:
        print(fv, None)    

#alist=['我们一起学猫叫','们','一','起','学','猫','叫']
#
#commentFile = open("allcomment.csv",'w+',encoding='utf-8',newline="")
#fieldnames = ['date','time','url','author', 'comment', 'emotion']
#writer = csv.DictWriter(commentFile, fieldnames=fieldnames)
#writer.writeheader()
#commentFile.close() 
#commentFile = open("allcomment.csv",'a',encoding='utf-8',newline="")
#writer = csv.DictWriter(commentFile, fieldnames=fieldnames)
#for comment in alist:
#    writer.writerow({'date':comment})
#commentFile.close()
 
    
#input_file = csv.DictReader(open("allcomment.csv",'r',encoding='utf-8',newline=''))
#for row in input_file:    
#    print(row['comment'])

pls = [3]
for i in range(10):
    pl = pls[:]
    pl += [3]
    print(pl)
    
file = open('afilename.txt','a',encoding='utf-8')

tlist = ['1','2','3','4','5','6','7','8','9']

for t in tlist:
    file = open('afilename.txt','w+',encoding='utf-8')
    file.write(t + '\n')    
    
for j in range(4,9):
    print(j)

string = 'abc-_'

print(re.search("_",string))


l1 = [0,0,0,0,0]
l2 = [1,2,3,4,5]
l3 = l1+l2
print(l3)
#import json
#
#EMO_NUM = 5
#EMO_LABEL = ['happiness','sadness','fear','anger','surprise']
#
#class Post:    
#    def get_post(self):    
#        self._post = { self._pid : {
#                    "ori" : self._ori,
#                    "new" : self._new,
#                    "emo" : self._emo_label.get_emo(),
#                    "detail": self._details.get_del()
#                }}
#        return self._post
#    
#    def __repr__(self):
#        return str(self.get_post())
#    
#    def __str__(self):
#        return str(self.get_post())
#    
#    def __init__(self, pid, ori, new, emo_label, details=None):
#        assert len(emo_label) == EMO_NUM, f"There should be {EMO_NUM} emotion labels."
#        self._pid = pid
#        self._ori = ori
#        self._new = new
#        self._emo_label = Emotion(emo_label)
#        self._details = Details(details)
#
#class Emotion:    
#    def get_emo(self):
#        emo_dict = {}
#        for i in range(len(EMO_LABEL)):
#            emo_dict[EMO_LABEL[i]] = self._emo_label[i]
#        return emo_dict
#    
#    def __init__(self, emo_label):
#        self._emo_label = emo_label
#        
#class Details:    
#    def get_del(self):
#        if self._del == None:
#            return None
#        self._dels = {
#                    "user" : "wahaha",
#                    "time" : 123,
#                    "likes" : 66,
#                    "shares" : 32,
#                    "comment": 777
#                }
#        return self._dels
#
#    def __init__(self, details=None):
#        self._del = details        
#        
#post1 = Post("post1","666", "555", [1,2,3,4,5])        
#post2 = Post("post2","777", "888", [1,2,3,4,5], 555)
#
#post_l = {}
#post_l.update(post1.get_post())
#post_l.update(post2.get_post())
#
#print(post1)
#print(post2)
#print(post_l)
#
#outFile = open("data/data_file.json",'w',encoding='utf-8')
#json.dump(post_l, outFile, ensure_ascii=False)

#import numpy as np
#
#data = [1.4, 2.0, 2.8, 3.4, 5.9, 6.0, 8.8, 9.4, 9.9]
#
#def normalize(data):
#    minb = np.min(data)
#    maxb = np.max(data)
#    rang = (maxb - minb) / 4
#    return (data - minb + rang/2) / (maxb - minb + rang)
#
#print(normalize(data))
#
#term_dict = {}
#
#infile = open("data/WeiBoCorpusBigramJiebaEngPos.txt",'r',encoding='utf-8').read().split('\n')
#leng = len(infile) - 1
#count = 0
#
#for terms in infile:
#    try:
#        term, pos, f, df = terms.split(' ',3)
#        term_dict[term] = [pos]
#        count += 1
#    except:
#        if count == leng:
#            print('completed')
#        else:
#            print('not complete')  
#
#for z in term_dict.items():
#    x, y = z
#    y += [1]
#    print(z)