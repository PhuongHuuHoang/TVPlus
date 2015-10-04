#!/usr/bin/python

##  This program is written by Phuong H. Hoang on October 03 2015 ##
##  POS tagging using HMM for Vietnamese documents                ##

# -*- coding: utf8 -*-
from __future__ import print_function
import codecs
import nltk
import os, re

class HMMPOS(object):

    def __init__(self):
        pass

    def f_readFile(self, fileName):
        f = codecs.open(fileName,'rb','utf-8')
        return f.read()

    def f_token(self,text):
        data = []
        tmp  = [] 
        text_token = nltk.word_tokenize(text)
        #print (text_token)
        for idx in range(len(text_token)):
            if "./CH" in text_token[idx]:
                tmp.append(tuple(('.','.')))
                data.append(tmp)
                tmp = []
            else:
                if "CH" not in text_token[idx]:           
                    if "/" in text_token[idx]:
                        tmp.append(tuple(text_token[idx].split('/')))
                    if ',' == text_token[idx]:
                        tmp.append(tuple((',',',')))
        #print (data)
        return data

    def f_loadFileProcess(self, mainPath, paths={}):
        subPaths = os.listdir(mainPath)
        for path in subPaths:
            pathDir = pDir = os.path.join(mainPath, path)
            if os.path.isdir(pathDir):
                paths.update(self.f_loadFileProcess(pathDir, paths))
            else:
                paths[path] = pathDir
        return paths

    def f_createTrainData(self, mainPath):
        paths = self.f_loadFileProcess(mainPath)
        labelled_sequences = []
        sentence = []
        tag_set = set()
        symbols = set()
        tag_re = re.compile(r'[*]|--|[^+*-]+')
        for file, pathFile in paths.items():
            if ".pos" in file and "test" not in file:
                text = self.f_readFile(pathFile)
                data_1_text = self.f_token(text)
                for idx_1 in range(0, len(data_1_text)):
                    for idx_2 in range(0,len(data_1_text[idx_1])):
                        word = data_1_text[idx_1][idx_2][0]
                        word = word.lower() 
                        tag = data_1_text[idx_1][idx_2][1]
                        #print (tag)
                        sentence.append((word,tag))
                        tag = tag_re.match(tag).group()
                        tag_set.add(tag)
                        symbols.add(word) 
                    labelled_sequences.append(sentence)
                    sentence = []                     
        #print (symbols)
        #print (tag_set)
        #print (labelled_sequences)
        return labelled_sequences, list(tag_set), list(symbols)

    def f_createTestData(self, mainPath):
        paths = self.f_loadFileProcess(mainPath)
        labelled_sequences = []
        sentence = []
        tag_set = set()
        symbols = set()
        tag_re = re.compile(r'[*]|--|[^+*-]+')
        for file, pathFile in paths.items():
            if ".pos" in file and "test" in file:
                text = self.f_readFile(pathFile)
                data_1_text = self.f_token(text)
                for idx_1 in range(0, len(data_1_text)):
                    for idx_2 in range(0,len(data_1_text[idx_1])):
                        word = data_1_text[idx_1][idx_2][0]
                        word = word.lower() 
                        tag = data_1_text[idx_1][idx_2][1]
                        #print (tag)
                        sentence.append((word,tag))
                        tag = tag_re.match(tag).group()
                        tag_set.add(tag)
                        symbols.add(word) 
                    labelled_sequences.append(sentence)
                    sentence = []                     
        #print (symbols)
        #print (tag_set)
        #print (labelled_sequences)
        return labelled_sequences, list(tag_set), list(symbols)
    
    def f_trainHMM(self):
        from nltk.probability import LidstoneProbDist
        print("...")
        print("... Training HMM POS tagging")
        labelled_sequences, tag_set,symbols = \
                            self.f_createTrainData(r"C:\\Python34\TVplus\HMM_POS_TVplus\Trainset-POS-1")
        trainer = nltk.tag.hmm.HiddenMarkovModelTrainer(tag_set,symbols)
        hmm = trainer.train_supervised(labelled_sequences[0:],\
                                       estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins))
        print ("... Finished training HMM POS tagging")
        print ("...")
        return hmm

    def f_testHMM(self):
        hmm = self.f_trainHMM()
        labelled_sequences, tag_set,symbols = \
                            self.f_createTestData(r"C:\\Python34\TVplus\HMM_POS_TVplus\Trainset-POS-1")
        print ("...")
        print ("... Testing HMM POS tagging")
        HMMtagger = hmm.test(labelled_sequences, verbose = True)
        print ("... Finished testingHMM POS tagging")
        print ("...")

    def f_run(self):
        self.f_testHMM()


    
          

def main():
    POS = HMMPOS()
    POS.f_run()

if __name__ == '__main__':
    main()
        
