import json
import numpy as np
import factorgraph as fg

PL = ["ori","new","emo","detail"]   # Post Label
EMO_NUM = 5
EL = ['happiness','sadness','fear','anger','surprise']  # Emotion Label

ALL_POST = 120
TRAIN_P = 100

data_file = open('data/data_file.json','r',encoding='utf-8')
data_dict = open('data/data_file2.json','r',encoding='utf-8')
emop_dict = open('data/emop_file.json','r',encoding='utf-8')

posts = json.load(data_file)
w_pot = json.load(data_dict)
emo_p = json.load(emop_dict)

g = fg.Graph()

i = 0
for postid, post in posts.items():
    if i >= ALL_POST:
        break
    
    labels = [label[0] for label in post[PL[1]]]
    emo = post[PL[2]]
    
    pots = [np.array([w_pot[label][5] if label in w_pot else 0.1 for label in labels]) for i in range(len(EL))]
    g.rv(postid, len(post['new']), labels=labels, pots=pots)
    factor = [w_pot[label][5] if label in w_pot else 0.1 for label in labels]
    g.factor([postid], potential=np.array(factor))
    
#    factors = [[w_pot[label][5],w_pot[label][5]] if label in w_pot else [0.1,0.1] for label in labels]
#    factors = [[1.0,1.0] if label in w_pot else [1.0,1.0] for label in labels]
    for j in range(len(EL)):        
        rv_name = postid+'_'+EL[j]
        if i < TRAIN_P:
            if emo[EL[j]] == 1:
                this = np.array([1.0, 0])
            else:
                this = np.array([0, 1.0])
        else:
            this = np.array([0.5, 0.5])
        g.rv(rv_name, 2, labels=['1','0'])        
        g.factor([rv_name], potential=this)
        
        factors = [[emo_p[label][j], 1-emo_p[label][j]] if label in emo_p else [0.5,0.5] for label in labels]
        
        g.factor([postid, rv_name], potential=np.array(factors))
        
    i += 1
    

iters, converged = g.lbp(normalize=True)
print('LBP ran for %d iterations. Converged = %r' % (iters, converged))
print()


#g.print_messages()
#print()


g.print_rv_marginals(normalize=True)
   
print('LBP ran for %d iterations. Converged = %r' % (iters, converged))