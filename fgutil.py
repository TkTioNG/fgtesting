# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 09:57:07 2019

@author: User
"""

# =============================================================================
#  Problem: Data Without Stopword Removal
# =============================================================================

import re
import json
import numpy as np
from math import log
from opencc import OpenCC
cc = OpenCC('t2s')
from jieba import posseg

ENG_DICT_FP = 'dict/eng_dict_2.txt'

# =============================================================================
# Filter Data Crawled From WeiBo
# =============================================================================
class filterWeiBo:
    
    def filterCell(self, value):
        '''
        Remove the unwanted content in the post that will not bring meaning 
        to the post and emotion analysis.
        
        Args:
            value (str)  Content of the post
            
        Returns:
            str  Clean content of the post
        '''
        value = cc.convert(value)
        # filter username tag from @ to a space
        newvalue = re.sub(u'@[^\s]+',u'',value)
        # filter away #hashtag#, in weibo from # to #
        newvalue = re.sub(u'#[^#]+#',u'',newvalue)
        # filter away tag that is to long in row.
        newvalue = re.sub(u'...全文',u'',newvalue)       
        # filter away tag that is to long in row.
        newvalue = re.sub(u'网页链接',u'',newvalue)    
        # filter away tag 超话
        newvalue = re.sub(u'超话',u'',newvalue)   
        # filter away @username的微博视频
        newvalue = re.sub(u'[^\s+]+的微博视频',u'',newvalue)       
        # filter away // forward clause
        newvalue = re.sub(u'//',u'',newvalue)         
        return newvalue
    
    def loadEngDict(self):
        '''
        Loading the English dictionary for the purpose of checking the
        english words/phrases/contents exist in a post.
        
        Returns:
            dict  items of the dict will be an unique english word
        '''
        dictFile = open(ENG_DICT_FP).read()
        eng_word = {}
        for word in dictFile.split('\n'):
            eng_word[word] = None
        return eng_word      
    
    def checkChinese(self, value):
        '''
        Searching of chinses characters in a post for the purpose of 
        checking the chinese words/phrases/contents exist in a post
        through the unicode allocated for chinese characters.
        
        Returns:
            bool  True (has chinese character), vice versa
        '''
        # Check for chinese characters
        if re.search("[\u4e00-\u9FFF]", value):
            return True
        return False
            
    def checkEnglish(self, value):
        '''
        Checking of english words in a post based on the preloaded dictionary.
        
        Returns:
            bool  True (has english words), vice versa
        '''
        newvalue = ""
        for char in value:
            if char.upper() not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                char = " "
            newvalue += char
        words = newvalue.split()
        for word in words:
            if word.upper() in self.eng_word:
                return True
        return False
    
    def __init__(self):
        '''
        Immediately load the English dictionary.
        '''
        self.eng_word = self.loadEngDict()


# =============================================================================
# calculating phrase-based TF-IDF of the overall WeiBo Data
# =============================================================================
class TFIDF_Exception(Exception):
   """Raised when exception happened LOL"""
   pass        
        
class TFIDF:
    def getTF(self, f):
        '''
        Get the term frequency in float.
        
        Args:
            f (int|float)  term frequency 
        
        Returns:
            float
        '''
        return float(f)
    
    def getIDF(self, df):
        '''
        Get the inverse-document frequency based on the document frequency.
        
        Args:
            df (int|float)  document frequency
            
        Returns:
            float
        '''
        return log(self.doc_count / (1 + float(df)))
    
    def getTFIDF(self, f_l, df_l):
        '''
        Get the tf-idf of each words/phrases with their repective
        term frequency(tf) and document frequency(df) in the sample data.
        
        Args:
            f_l (list of int|float)  frequency list
            d_l (list of int|float)  document frequency list
            
        Returns:
            list of float
        '''
        ti_l = []
        
        if len(f_l) != len(df_l):
            raise TFIDF_Exception('Length of term frequency list is not the same as document frequency list, {}, {}'.format(len(f_l), len(df_l)))
            
        for i in range(len(f_l)):   
            tf = self.getTF(f_l[i]) 
            idf = self.getIDF(df_l[i])
            ti_l.append([f_l[i], tf, df_l[i], idf, log(tf * idf)])
        ti_l = self.normalize(ti_l)    
            
        return ti_l
    
    def normalize(self, ti_l):
        '''
        Normalize the tf-idf of each words/phrases in the sample data.
        
        Args:
            ti_l (list of float)  list of tf-idf of the words
            
        Returns:
            list of float
        '''
        data = [dt[4] for dt in ti_l]
        minb = np.min(data)
        maxb = np.max(data)
        # To approximate the quartile distribution of the tf-idf
        rang = (maxb - minb) / 4
        # Range the tf-idf in the range of 0.1-0.9
        norm = (data - minb + rang/2) / (maxb - minb + rang)
        for i in range(len(ti_l)):
            ti_l[i][4] = norm[i]
        return ti_l
        
    def __init__(self, doc_count):
        '''
        Initialize the document count for the purpose of
        calculating document frequency.
        '''
        self.doc_count = doc_count
        
        
        
def addWordsPos(dictionary, words):
    '''
    Form a dictionary as a collection of words and phrases from
    the sample data for the ease of calculation on the 
    tf and df of each words/phrases.
    
    Args:
        dictionary (dict) The dictionary object
        words (list [pos (str), tf (int|float), df (int|float)]) 
        
    Returns:
        dict
    '''
    for word in words:
        if(dictionary.get(word[0])):
            x, y, z = dictionary[word[0]]
            dictionary[word[0]] = [x, y+1, z]
        else:
            dictionary[word[0]] = [word[1], 1, 0]
            
    x_func = lambda x : [y[0] for y in x]        
#    print(words)
#    print(x_func(words))
    for word in list(dict.fromkeys(this for this in x_func(words))):
#        print(word)
#        print(dictionary[word])
        x, y, z = dictionary[word]
        dictionary[word] = [x, y, z+1]


# =============================================================================
# N-gram extraction of the WeiBo data crawled
# =============================================================================
class NgramException(Exception):
   """Raised when the n value is smaller than 1"""
   pass

class Ngram:    
    n = 2
    stopw_all = json.load(open('dict/stopwords_all.json','r',encoding='utf-8'))
    stopw_zh = json.load(open('dict/stopwords_zh.json','r',encoding='utf-8'))
    stopw_en = json.load(open('dict/stopwords_en.json','r',encoding='utf-8'))
    
    def change_n(self, n):
        if (n >= 1):
            self.n = n
        else:
            raise NgramException('n should not less than 1. The value of n was: {}'.format(n))
        
    def __init__(self, n=2, eng2 = True):
        '''
        Args:
            n (int, 2) ngrams
            eng2 (bool, True) segment for english
        '''
        self.n = n 
        self.eng2 = eng2
                
    def alSplit(self, sentence):
        '''
        Split up sentences ans punctuation in a post with the concerns of 
        english words
        
        Args:
            sentence (str) post contents
            
        Returns:
            list of str
        '''
        return list(filter(None, re.split("\s+|[^\u4e00-\u9FFFA-Za-z']+|[！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.]+",sentence)))
        
    def cnSplit(self, sentence):
        '''
        Split up sentences ans punctuation in a post without the concerns of 
        english words only chinese characters
        
        Args:
            sentence (str) post contents
            
        Returns:
            list of str
        '''
        return re.split("\s+|[^\u4e00-\u9FFF]+|[！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.]+",sentence)
    
    def stopwRemove(self, doc, cn = True, en = True):
        '''
        Remove stopwords from the sentence.
        
        Args:
            doc (str|file location)
            cn (bool)
            en (bool)
            
        Returns:
            str
        '''
        stopw_l = self.stopw_all
        if cn and not en:
            stopw_l = self.stopw_zh
        elif not cn and en:
            stopw_l = self.stopw_en
        
        if type(doc) == str:
            if doc.lower().endswith(('.json')):
                data = json.load(open(doc,'r',encoding='utf-8'))
                new_data = {}
                for word, value in data.items():
                    if word.lower() not in stopw_l:
                        new_data.update({word.lower() : value})
                json.dump(new_data,open(doc,'w',encoding='utf-8'),ensure_ascii=False)
            elif doc.lower().endswith(('.txt')):
                sentence = open(doc,encoding='utf-8').read().split('\n')
                stopw = []
                clean = []
                for sen in sentence:
                    word = sen.split(' ', 1)[0].lower()
                    if word in stopw_l:
                        stopw.append(sen)
                    else :
                        clean.append(sen)
                with open("outcome/featSelEngClean.txt",'w',encoding='utf-8') as outFile:
                    for sen in clean:
                        outFile.write(f'{sen}\n')
                with open("outcome/featSelEngStopw.txt",'w',encoding='utf-8') as outFile:
                    for sen in stopw:
                        outFile.write(f'{sen}\n')                    
        else:
            list_of_words = []
            for word in doc:
                if word[0].lower() in stopw_l:
                    doc.remove(word)
                else:
                    list_of_words += [word[0]]
            return doc, list_of_words        
            # print(stopw_list)    
        pass
    
    def cnNgram(self, sentence):
        '''
        Segment the words in a post based on the Ngrams predefined in the class
        
        Args:
            sentence (str) post contents
            
        Returns:
            list of str
        '''
        if self.eng2 == True:
            sentences = self.alSplit(sentence)
        else:
            sentences = self.cnSplit(sentence)
#        print(sentences)
        cnNgram = []
        for sen in sentences:
            for j in range(len(sen) - (self.n-1)):
                word = sen[j]
                for k in range(1,self.n):
                    word += sen[j+k]      
        return cnNgram
    
    def cnNgramPos(self, sentence):
        '''
        Segment the words along with the pos tagging in a post based on 
        the Ngrams predefined in the class
        
        Args:
            sentence (str) post contents
            
        Returns:
            list of str
        '''
        if self.eng2 == True:
            sentences = self.alSplit(sentence)
        else:
            sentences = self.cnSplit(sentence)
        
        cnNgramJiebaPos = []
        list_of_words = []
        for sen in sentences:
            jieba_cut = posseg.cut(sen)
            for word, flag in jieba_cut:
                if word.lower() not in list_of_words:
                    cnNgramJiebaPos += [[word.lower(), flag]]
                    list_of_words += [word.lower()]
            # print(jieba_cut)
        return cnNgramJiebaPos


# =============================================================================
# Data Structure of the post with emotion labels
# =============================================================================
EMO_NUM = 5
EMO_LABEL = ['happiness','sadness','fear','anger','surprise']

class Post:    
    def get_post(self):    
        self._post = { self._pid : {
                    "ori" : self._ori,
                    "new" : self._new,
                    "emo" : self._emo_label.get_emo(),
                    "detail": self._details.get_del()
                }}
        return self._post
    
    def __repr__(self):
        return str(self.get_post())
    
    def __str__(self):
        return str(self.get_post())
    
    def __init__(self, pid, ori, new, emo_label, details=None):
        assert len(emo_label) == EMO_NUM, f"There should be {EMO_NUM} emotion labels."
        self._pid = pid
        self._ori = ori
        self._new = new
        self._emo_label = Emotion(emo_label)
        self._details = Details(details)

class Emotion:    
    def get_emo(self):
        emo_dict = {}
        for i in range(len(EMO_LABEL)):
            emo_dict[EMO_LABEL[i]] = self._emo_label[i]
        return emo_dict
    
    def __init__(self, emo_label):
        self._emo_label = emo_label
        
class Details:    
    def get_del(self):
        if self._del == None:
            return None
        self._dels = {
                    "user" : "wahaha",
                    "time" : 123,
                    "likes" : 66,
                    "shares" : 32,
                    "comment": 777
                }
        return self._dels

    def __init__(self, details=None):
        self._del = details 