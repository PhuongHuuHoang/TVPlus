#! /user/bin/env python
"""
    It is to classify texts using Naive Bayes classification.
    
    Task: On October 03 2015, Real Madrid makes a ceremony to celebrate
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
        return ''.join(ch for ch in s if ch not in exclude)

    def f_token(self,text):
        """
        This function is to tokenize text
        Input: text
        Output: token text
        """
        import nltk
        text = self.f_rmPunc(text)
        text = text.lower()
        return nltk.word_tokenize(text)
        
    def f_countWords(self,text):
        """
        This function is to count words after tokenization
        Input: text
        Output: {word : word count}
        """
        word_counts = {}
        for word in text:
            word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts
    
    def f_fileProcess(self, mainPath, paths = {}):
        """
         This function is to process count files and categorize file into class
         Input: main path of traning texts
         Output: {file : file's location}
        """
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
        self.T_c = {"pos":{},"neg":{}}
        conProb = {"pos":{},"neg":{}}
        paths = self.f_fileProcess(path_data)         
        for f in paths.items():
            if "pos" == (f[0][0:3]):
                category = "pos"
            else:
                category = "neg"
            priors[category] += 1
            text = open(f[1]).read()
            words = self.f_token(text)
            word_counts = self.f_countWords(words)
            for word, count in list(word_counts.items()):
                if word not in V:
                    V[word] = 0.0  
                if word not in self.T_c[category]:
                    self.T_c[category][word] = 0.0
                V[word] += count
                self.T_c[category][word] += count
        
        for category, count in list(self.T_c.items()):
            #print (category)
            for word in self.T_c[category]:
                #print ("test")
                conProb[category][word] = (self.T_c[category][word]+1)/(sum(self.T_c[category].values())+ len(V))
        #print (conProb)           
        N = sum(priors.values())
        for category, count in priors.items():
            tmp = (priors[category] / N)
            priors[category] = tmp
        return V,priors,conProb

    def f_testNB(self,testPath, trainPath):
        "Test NB classification"
        V, priors,conProb = self.f_trainNB(trainPath)
        test_text = open(testPath).read()      
        score = {}
        for category, count in list(self.T_c.items()):
            score[category] = math.log(priors[category])
        words = self.f_token(test_text)
        word_counts = self.f_countWords(words)
        for w, count in list(word_counts.items()):
            for category, count in list(self.T_c.items()):
                if w not in self.T_c[category]:
                    continue
                score[category] +=  math.log(conProb[category][w])
        if score['neg'] >= score['pos']:
            print ("... Ronaldo Fan")
        else:
            print ("... Not Ronaldo Fan")
        
    def f_run(self):
        "This function is for running the program"
        self.f_testNB(r"C:\\Python34\TVplus\NBclass_TVplus\data\testData\test4.txt", \
                          r"C:\\Python34\TVplus\NBclass_TVplus\\data\trainData")

def main():
    NBclass = NBClassifier()
    NBclass.f_run()

if __name__ == '__main__':
    main()
