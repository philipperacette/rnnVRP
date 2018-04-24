
# coding: utf-8

# In[1]:


#!pip install -U tqdm


# In[2]:


import os
import pandas as pd
import numpy as np
import random
from random import shuffle
import matplotlib.pyplot as plt
from pandas.tools.plotting import table
from tqdm import tqdm_notebook, tnrange


# In[3]:


path = "./data/sol.out"


# In[4]:


shift_schedule = pd.read_csv(path, delim_whitespace=True, header=-1)


# In[5]:


shift_schedule.columns = ['employe', 'poste', 'quantite', 'tâche', 'num_rota', 'debut', 'heure_debut', 'fin', 'heure_fin', 'parentheses']


# In[6]:


#sont gardées seulement les rangées pour les rotations d'employés (pas de jours de congés ou de formation inclus)
shift_work = shift_schedule[shift_schedule['tâche'] == 'TRP']


# In[7]:


#on garde les colonnes pertinentes pour le projet
shift_assignments = shift_work[['employe', 'num_rota']]


# In[ ]:


#permet de déterminer les targets (rotations suivantes) pour chaque rotation lorsqu'en position initiale 
shift_assignments['next'] = None
shift_assignments['next2'] = None   


# In[9]:


shift_assignments.index = range(23682)


# In[10]:


shifts = np.array(shift_assignments)


# In[11]:


#shifts


# In[12]:


for i in range(shifts.shape[0] - 1):
    shifts[i, 2] = shifts[i+1, 1]
for i in range(shifts.shape[0] -2):
    shifts[i, 3] = shifts[i+2, 1]


# In[13]:


#shifts


# In[14]:


#pour énumérer les rotations uniques, utile pour la création de one-hot vectors
code_rota = list(set(shift_assignments['num_rota']))
code_to_ix = {code:i for i,code in enumerate(code_rota)}


# In[15]:


#conversion des codes en nombres entiers (permet embedding et one hot vector)
shifts_filt = [shifts[s, :] for s in range(len(shifts)-2) if shifts[s, 0] == shifts[s+2, 0]]
inputs_0 = [code_to_ix[code] for code in [i[1] for i in shifts_filt]]
inputs_1 = [code_to_ix[code] for code in [i[2] for i in shifts_filt]]
inputs_2 = [code_to_ix[code] for code in [i[3] for i in shifts_filt]]
inputs_whole = list(zip(inputs_0, inputs_1, inputs_2))
#print(inputs_whole)


# In[16]:


shuffle(inputs_whole)
#print(inputs_whole)


# In[17]:


inputs_train = inputs_whole[0:15547]
inputs_test = inputs_whole[15548:19434]


# In[37]:


hidden_size = 100
learning_rate = 1e-2
rota_size = len(code_rota)
nb_epochs = 100
minibatch_size = len(inputs_whole)//200
losses_train = []
losses_test = []


# In[38]:


Wxh = np.random.randn(hidden_size, rota_size)*0.01
Whh = np.random.randn(hidden_size, hidden_size)*0.01
Why = np.random.randn(rota_size, hidden_size)*0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((rota_size, 1))


# In[39]:


# Source: https://gist.github.com/karpathy/d4dee566867f8291f086
# Post: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
#le code est largement inspiré de cette source, mais a été adapté pour les besoins de notre projet.
def lossFun(inputs, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    #xs, hs, ys, ps = [], [], [], []
    hs[-1] = np.copy(hprev)
    loss = 0
    
    xs[0] = np.zeros((rota_size, 1))
    xs[0][inputs[0]] = 1
    
    #propagation forward
    for t in range(2):
        target = np.zeros((rota_size, 1))
        target[inputs[t+1]] = 1
        
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(np.dot(ps[t].T,target))
        
        #input suivant assigné selon l'indice correspondant à la probabilité maximale
        xs[t+1] = np.zeros((rota_size, 1))
        ix = np.argmax(ps[t])
        xs[t+1][ix] =1
        
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    
    #backpropagation avec BTT
    for t in range(1, -1, -1):
        dy = np.copy(ps[t])
        dy[inputs[t+1]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
    
    #pour éviter explosion des gradients
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)
    return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-2]

#pour évaluer fonction de perte de l'ensemble test sans ajustement de gradient
def justLoss(inputs, hprev):
    xs, hs, ys, ps = {}, {}, {}, {}
    #xs, hs, ys, ps = [], [], [], []
    hs[-1] = np.copy(hprev)
    loss = 0
    
    xs[0] = np.zeros((rota_size, 1))
    xs[0][inputs[0]] = 1
    
    for t in range(2):
        target = np.zeros((rota_size, 1))
        target[inputs[t+1]] = 1
        
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh)
        ys[t] = np.dot(Why, hs[t]) + by
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += -np.log(np.dot(ps[t].T,target))
        
        xs[t+1] = np.zeros((rota_size, 1))
        ix = np.argmax(ps[t])
        xs[t+1][ix] =1
        
    return loss
        


# In[40]:


mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by)

for epoch in tnrange(nb_epochs, desc='processing epoch'):    
    
    #afin de considérer une mise à jour par minilot
    for i in tnrange(0, len(inputs_train), minibatch_size, desc='inputs_whole', leave=False):
        inputs_select = inputs_train[i:(i+minibatch_size)]
        loss, dWxh, dWhh, dWhy, dbh, dby = [0, 0, 0, 0, 0, 0]
        hprev_0 = np.zeros((hidden_size,1))
        hprev = np.zeros((hidden_size,1))
        
        for j in range(len(inputs_select)):
            res = [q/len(inputs_select) for q in lossFun(inputs_select[j], hprev_0)]
            loss += res[0]
            dWxh += res[1]
            dWhh += res[2]
            dWhy += res[3]
            dbh += res[4]
            dby += res[5]
            hprev += res[6]
        
        #mise à jour des gradients
        for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby], [mWxh, mWhh, mWhy, mbh, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
            
    print(loss)
    losses_train.append(loss)

    loss = 0
    hprev_0 = np.zeros((hidden_size,1))
    hprev = np.zeros((hidden_size,1))
     
    #pour les fonctions de perte de test
    for l in range(len(inputs_test)):
        res = [k/len(inputs_test) for k in justLoss(inputs_test[l], hprev_0)]
        loss += res[0]
    print(loss)
    losses_test.append(loss)


# In[52]:


losses_train = np.squeeze(losses_train)


# In[53]:


losses_train.shape = (100, 1)


# In[54]:


#losses_train


# In[55]:


#losses_test


# In[56]:


get_ipython().run_line_magic('pinfo', 'pd.DataFrame')


# In[57]:


train_table = pd.DataFrame({'train': losses_train.flatten(), 'test': np.array(losses_test).flatten()})[:]
plt.savefig('table.png', transparent=True)

print(train_table.to_latex())
# In[45]:


train_table.plot(title = 'fonction de perte par epoch, entraînement et test')
plt.savefig('magni.png', transparent=True)

