# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:37:51 2019

@author: bananamilk
"""

import nltk
from nltk.corpus import brown
import csv



    

def get_prob(data_train):
    pi_prob = {}
    ob_prob = {}   
    for sent in data_train:
        pi_prob[sent[0][1]] = pi_prob.get(sent[0][1], 0) + 1
        for word, tag in sent:
            if tag not in ob_prob:
                ob_prob[tag] = {}
                ob_prob[tag][word] = ob_prob[tag].get(word, 0) + 1
            else:
                ob_prob[tag][word] = ob_prob[tag].get(word, 0) + 1
                   
    for tag in ob_prob:
        total = sum(ob_prob[tag].values())
        for word in ob_prob[tag]:
            ob_prob[tag][word] = ob_prob[tag][word]/total
    
    for tag in ob_prob:
        if tag not in pi_prob:
            pi_prob[tag] = 0
    tot = sum(pi_prob.values())
    for tag in pi_prob:
        pi_prob[tag] = pi_prob[tag]/tot
            
    trans_prob = {}        
    for sent in data_train:
        for idx in range(len(sent)-1): 
            cur_tag = sent[idx][1]
            if cur_tag not in trans_prob:
                trans_prob[cur_tag] = {} 
                nex_tag = sent[idx+1][1]
                trans_prob[cur_tag][nex_tag] = trans_prob[cur_tag].get(nex_tag, 0) + 1
            else:
                nex_tag = sent[idx+1][1]
                trans_prob[cur_tag][nex_tag] = trans_prob[cur_tag].get(nex_tag, 0) + 1
    
    for cur_tag in trans_prob:
        total = sum(trans_prob[cur_tag].values())
        for nex_tag in trans_prob[cur_tag]:
            trans_prob[cur_tag][nex_tag] = trans_prob[cur_tag][nex_tag]/total
              
    return ob_prob, trans_prob, pi_prob



def VITERBI(ob_prob, trans_prob, pi_prob, obs, state):
    T = len(obs)
    N = len(state)
    viterbi = [[0 for _ in range(T)] for _ in range(N)]
    backpointer = [[0 for _ in range(T)] for _ in range(N)]

    for s in range(N):
        if obs[0] not in ob_prob[state[s]]:
            b_o = 0
        else:
            b_o = ob_prob[state[s]][obs[0]]
        viterbi[s][0] = pi_prob[state[s]]*b_o

        
    for t in range(1, T):
        for s in range(N):
            for i in range(N):
                if state[i] not in trans_prob:
                    continue
                else:
                    if state[s] not in trans_prob[state[i]]:
                        continue
                    else:
                        a = trans_prob[state[i]][state[s]]
                if obs[t] not in ob_prob[state[s]]:
                    continue
                else:
                    b_o = ob_prob[state[s]][obs[t]]

                if viterbi[i][t-1]*a*b_o > viterbi[s][t]:
                    viterbi[s][t] = viterbi[i][t-1]*a*b_o
                    backpointer[s][t] = i
    bestpathprob = 0
    bestpathpointer = 0
    for s in range(N):
        if viterbi[s][T-1] > bestpathprob:
            bestpathprob = viterbi[s][T-1]
            bestpathpointer = s
            
    bestpath = []
    cur = bestpathpointer
    for i in range(T-1, -1, -1):
        bestpath.append(cur)
        cur = backpointer[cur][i]
    bestpath = [state[i] for i in bestpath]
    return bestpath[::-1], bestpathprob
        


   

def evaluate(data_test, data_eval, ob_prob, trans_prob, pi_prob, state):

    debug =[]
    predict = []
    for sent in data_test:
        bestpath, bestpathprob = VITERBI(ob_prob, trans_prob, pi_prob, sent, state)
        predict.append(bestpath)
    
    total = 0
    err = 0
    for i, sent in enumerate(data_eval):
        for j, (word, gold) in enumerate(sent):
            total += 1
            if gold != predict[i][j]:
                err += 1
                debug.append((word, gold, predict[i][j]))

    #with open(r'.\error.csv', mode = 'w') as error:
       # writer = csv.writer(error, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
       # writer.writerow(['word', 'gold tag', 'predict tag'])
       # writer.writerows(debug)

   
    if total:
        erro_rate = err/total
    else:
        erro_rate = 0
                


    return erro_rate


def main():
    # n_total_sents = 57340
    # n_total_words = 1161192

    n_train = 50000
    n_test= 500
    
    data_train = brown.tagged_sents()[:n_train]
    data_eval = brown.tagged_sents()[n_train:n_train+n_test]
    #data_eval = brown.tagged_sents()[:n_test]
    data_test = brown.sents()[n_train:n_train+n_test]
    #data_test = brown.sents()[:n_test]
    
    
    ob_prob, trans_prob, pi_prob = get_prob(data_train)
    state = list(ob_prob.keys())
    error_rate = evaluate(data_test, data_eval, ob_prob, trans_prob, pi_prob, state)
    print('error rate  is {}'.format(error_rate))

    
if __name__ == '__main__':
    main()    
        
        
    
    
    
    
    
            
            
        
