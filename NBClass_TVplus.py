#! /user/bin/env python
"""
    This program is written by Phuong H. Hoang in order to apply
    into TVplus, data mining position on October 03 2015.
    It is to classify texts using Naive Bayes classification.
    
    Task: On October 03 2015, Real Madrid make a ceremony to celebrate
    all-time goalscorers for Cristian Ronaldo. They posted some pictures in
    the event on Facebook and get comments from followers about Ronaldo.
    Let's classify who are fans of Ronaldo and who are not.
"""

from __future__ import division
import re
import string
import math
import os, os.path

class NBClassifier(object):
    "This class is to classify texts using Naive Bayes classification"

    def __init__(self):
        pass
    
    def f_rmPunc(self,s):
        "This function is to remove punctuation,..."
        exclude = set(string.punctuation)
        return ''.join(ch for ch in s if ch not in exclude )

    def f_token(self,text):
        "This function is to tokenize text"
        text = self.f_rmPunc(text)
        text = text.lower()
        return re.split("\W+", text)

    def f_countWords(self,words):
        "This function is to count words after tokenization"
        wc = {}
        for word in words:
            wc[word] = wc.get(word, 0.0) + 1.0
        return wc
    
    def f_fileProcess(self, mainPath, paths = {}):
        "This function is to process count files and categorize file into class"
        subPaths = os.listdir(mainPath)
        for path in subPaths:
            pathDir = pDir = os.path.join(mainPath, path)
            if os.path.isdir(pathDir):
                paths.update(self.f_fileProcess(pathDir, paths))
            else:
                paths[path] = pathDir
        return paths

    def f_trainNB(self, path_data):
        "Train NB classification"
        V = {}
        priors = {"pos":0.,"neg":0.}
        self.word_counts = {"pos":{},"neg":{}}
        self.docs = []
        paths = self.f_fileProcess(path_data)         
        for f in paths.items():
            if "pos" == (f[0][0:3]):
                category = "pos"
            else:
                category = "neg"
            self.docs.append((category, f[0]))
            priors[category] += 1
            text = open(f[1]).read()
            words = self.f_token(text)
            counts = self.f_countWords(words)
            for word, count in list(counts.items()):
                if word not in V:
                    V[word] = 0.0  
                if word not in self.word_counts[category]:
                    self.word_counts[category][word] = 0.0
                V[word] += count
                self.word_counts[category][word] += count
        N = sum(priors.values())
        for category, count in priors.items():
            tmp = (priors[category] / N)
            priors[category] = tmp
        return V,priors

    def f_testNB(self,testPath, trainPath):
        "Test NB classification"
        V, priors = self.f_trainNB(trainPath)
        new_doc = open(testPath).read()
        log_prob_pos = 0.
        log_prob_neg = 0.
        words = self.f_token(new_doc)
        counts = self.f_countWords(words)
        for w, cnt in list(counts.items()):
            if w not in V:
                continue
            p_word = V[w] / sum(V.values())
            p_w_given_pos = self.word_counts["pos"].get(w, 0.0) / sum(self.word_counts["pos"].values())
            p_w_given_neg = self.word_counts["neg"].get(w, 0.0) / sum(self.word_counts["neg"].values())

            if p_w_given_pos > 0:
                log_prob_pos += math.log(cnt * p_w_given_pos / p_word)
            if p_w_given_neg > 0:
                log_prob_neg += math.log(cnt * p_w_given_neg / p_word)
        pos = math.exp(log_prob_pos + math.log(priors["pos"]))
        neg = math.exp(log_prob_neg + math.log(priors["neg"]))
        if pos >= neg:
            print ("Ronaldo fan")
        else:
            print("Not Ronaldo fan")
        
    def f_run(self):
        "This function is for running the program"
        self.f_testNB(r"C:\\Python34\TVplus\NBclass_TVplus\data\testData\test1.txt", \
                          r"C:\\Python34\TVplus\NBclass_TVplus\\data\trainData")

def main():
    NBclass = NBClassifier()
    NBclass.f_run()

if __name__ == '__main__':
    main()


