# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:37:51 2019

@author: bananamilk
"""

import nltk
from nltk.corpus import brown
import csv




def most_likely_train(data_train):
    dic = {}
    for sent in data_train:
        for word, tag in sent:
            if word not in dic:
                dic[word] = {}
                dic[word][tag] = dic[word].get(tag, 0) + 1
            else:
                dic[word][tag] = dic[word].get(tag, 0) + 1
        most_likely = {}
        for word in dic:
            m = sorted(dic[word].items(), key=lambda d: d[1], reverse=True)[0][0]
            most_likely[word] = m
    return most_likely

def most_likely_test(most_likely, data_eval):

    erro_known = 0
    known = 0
    erro_unknown = 0
    unknown = 0
    erro_rate_known = 0
    erro_rate_unknown = 0
    debug =[]
    for sent in data_eval:
        for word, gold in sent:
            if word in most_likely:
                known += 1
                predict = most_likely[word]
                if predict != gold:
                    erro_known += 1
    
            else:
                unknown += 1
                if word[-1] == 's':
                    predict = 'NNS'
                else:
                    predict = 'NN'            
                if predict != gold:
                    erro_unknown += 1
                    debug.append((word, gold, predict))
    with open(r'.\error.csv', mode = 'w') as error:
        writer = csv.writer(error, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['word', 'gold tag', 'predict tag'])
        writer.writerows(debug)

   
    if known:
        #print(known)
        erro_rate_known = erro_known/known
    if unknown:
        #print(unknown)
        erro_rate_unknown = erro_unknown/unknown
                


    return erro_rate_known, erro_rate_unknown


def main():
    # n_total_sents = 57340
    # n_total_words = 1161192

    n_train = 50000
    n_test= 500
    
    data_train = brown.tagged_sents()[:n_train]
    data_eval = brown.tagged_sents()[n_train:n_train+n_test]
    
    most_likely = most_likely_train(data_train)
    erro_rate_known, erro_rate_unknown = most_likely_test(most_likely, data_eval)
    print('error rate onknown words is {}, on unknown words is {}'.format(erro_rate_known, erro_rate_unknown))

    
if __name__ == '__main__':
    main()    
        
        
    
    
    
    
    
            
            
        
