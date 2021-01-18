#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
from pandas import DataFrame as df
from collections import defaultdict

# function use to calculate the probability of emission and transition 
def prob(p_w, p_total):  
    new_list = []
    prob = []
    for i in p_w:
        for j in p_total:
            if i[0][0] == j[0]:
                new_list.append([i, j[1]])
    p = [k[0][1] / k[1] for k in new_list]
    prob = list(zip([k[0][0] for k in new_list], p))
    return prob

# function use to transfer the input data into the format we want
def observation(test_file_name):
    obs = []
    with open(test_file_name, 'r', encoding='utf8') as f:
        cont = True
        temp = []
        sentence = []
        new_sentence = []
        while cont:
            cont = f.readline()  # only want the pos of each line
            temp.append(cont)
            if cont == '\n':
                sentence.append(temp)
                temp = []
    for i in sentence:
        temp1 = [j.strip('\n') for j in i]
        new_sentence.append(temp1)
    for j in new_sentence:
        j.pop()
        obs.append(j)
    return obs

# function use to transfer the output data into the required foramt
def output(result):
    with open('result.pos', 'w+') as f:
        for i in result:
            for j in range(len(i)):
                #if i[j] == '\n':
                     #f.write('\n')
                    #continue
                if i[j][1] == None:
                    i[j] = list(i[j])
                    i[j][1] = 'None'
                f.write(str(i[j][0]))
                f.write('\t')
                f.write(str(i[j][1]))
                f.write('\n')
            f.write('\n')

# load the training data and seperate them into token and pos
train_data = np.loadtxt('POS_train.pos', delimiter='\t', dtype=str, comments='\n')

train_words = np.ndarray.flatten(np.split(train_data, [1, 1], axis=1)[0]).tolist()
train_words = [i.lower() for i in train_words]

train_pos = np.ndarray.flatten(np.split(train_data, [1, 1], axis=1)[2]).tolist()
words_pos = list(zip(train_pos, train_words))


trans_prob = defaultdict(lambda: 0.0)
emi_prob = defaultdict(lambda: 0.0)
pos_w = pd.value_counts(words_pos)
df0 = pd.DataFrame(pos_w)
emi_pos_w_count = list(zip(df0.index.values.tolist(), [pos_w[i] for i in range(len(pos_w))]))
emi_pos_count = pd.value_counts([i[0] for i in words_pos])
df1 = pd.DataFrame(emi_pos_count)
emi_pos_list = list(zip(df1.index.values.tolist(), [i for i in emi_pos_count]))
no_dup_words = [x[0][1] for x in emi_pos_w_count]

#calcualte the emission probability 
emi_prob = prob(emi_pos_w_count, emi_pos_list)
emi_matrix = pd.DataFrame(0, index=list(set(train_words)), columns=list(set(train_pos)))
# transfer the emission probability from dict to dataframe
for x in emi_prob:
    emi_matrix.loc[x[0][1], x[0][0]] = x[1]

#read the data sentence by sentence
with open('POS_train.pos', 'r', encoding='utf8') as f:
    cont = True
    temp = []
    sentence = []
    new_sentence = []
    while cont:
        cont = f.readline().split('\t')[-1]  # only want the pos of each line
        temp.append(cont)
        if cont == '\n':
            sentence.append(temp)
            temp = []
            
# new_sentence is the list of pos with sentence seperated
for i in sentence:
    temp1 = [j.strip('\n') for j in i]
    temp1.pop()
    new_sentence.append(temp1)
    
# put the pos pairs of each sentence into a list
pos_pairs = []
for i in new_sentence:
    for j in range(len(i) - 1):
        pos_pairs.append((i[j], i[j + 1]))
        
# count the number of each pos pairs
pos_pairs_c = pd.value_counts(pos_pairs)
df3 = pd.DataFrame(pos_pairs_c)
pos_pairs_count = list(zip(df3.index.values.tolist(), [pos_pairs_c[i] for i in range(len(pos_pairs_c))]))

# count the number with the same first element of pos pairs
p = pd.value_counts([x[0] for x in pos_pairs])
df4 = pd.DataFrame(p)
p_count = list(zip(df4.index.values.tolist(), [p[i] for i in range(len(p))]))

#calculate the transition probability 
trans_prob = prob(pos_pairs_count, p_count)
trans_matrix = pd.DataFrame(0, index=list(set(train_pos)), columns=list(set(train_pos)))

# transfer the transition probability from dict to dataframe
for x in trans_prob:
    trans_matrix.loc[x[0][0], x[0][1]] = x[1]

temp2 = [x[0] for x in new_sentence]
temp3 = pd.value_counts(temp2)
ss = []
df5 = pd.DataFrame(temp3)
not_init = list(set(x for x in set(train_pos))-set(df5.index.values.tolist()))

ss = [x for x in temp3] + [0]* len(not_init)
temp4 = [x/(len(new_sentence)) for x in ss]
pos_list = df5.index.values.tolist() + not_init

# calcualte the initial probability
init_prob = dict(zip(pos_list, [temp4[i] for i in range(len(temp4))]))


# In[3]:


def viterbi(obs, init_prob, trans_matrix, emi_matrix):
    result = []
    
    v = defaultdict(dict)
    backpointer = defaultdict(dict)
    
    pos = list(trans_matrix.columns)
    P = len(trans_matrix)
    
    for n in range(len(obs)):
        sentence = obs[n] # get each sentence out
        sentence = [i.lower() for i in sentence] # transfer all the token as lower case
        W = len(sentence)        
        
        for p in pos:
            for l in sentence:
                if l not in no_dup_words:
                    # for the token not in our train set, suppose their emi probability as uniform distribution
                    emi_matrix.loc[l, p] = 1/len(pos) 
            
            v[0][p] = init_prob[p]*emi_matrix.loc[sentence[0],p]      
        
        def argmax(w,p):            
            argmax_pre_pos, max_prob = 0, 0
            for s in pos:
                temp = v[w-1][s]*trans_matrix.loc[s,p]*emi_matrix.loc[sentence[w],p] 
                if temp > max_prob:
                    max_prob = temp
                    argmax_pre_pos = s
            return  max_prob, argmax_pre_pos                      
         
        for g in range(1, W):
            for j in pos:
                max_prob, argmax_pre_pos = argmax(g,j)
                v[g][j]= max_prob
                backpointer[g][j] = argmax_pre_pos    
            
        max_prob_f_pos, max_prob = None, 0               
        for s in pos:          
            if v[W - 1][s] > max_prob:
                max_prob_final_state = s
                max_prob = v[W - 1][s]
      
        best_path = [max_prob_final_state]
        for t in range(W - 1, 0, -1):
            try:
                prev_pos = backpointer[t][best_path[-1]]
                best_path.append(prev_pos)
            except:
                best_path.append(None)
                
        best_path = list(reversed(best_path))
        result.append(list(zip(sentence, best_path)))
        
    return result


if __name__ == '__main__':
    tag = viterbi(observation('POS_test.words'), init_prob, trans_matrix, emi_matrix)
    output(tag)


# In[ ]:




