import os
import json
import pandas as pd
import gensim
from gensim import corpora
from pprint import pprint
import pickle


from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec


'''
Configuaration for the trainig
'''
sg = 1 # if you want to train with CBOW, make it 0
window = 4
size = 300
min_count = 2
workers = 8
iters = 100
sample = 0.01

checkpoint = False

os.makedirs("results", exist_ok=True)


#os.makedirs("", exist_ok=True)  

# Call back funtion for saving the model after every epoch 
class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

        os.makedirs(self.path_prefix, exist_ok=True)

    def on_epoch_end(self, model):
        
        saved = "./checkpoints/epoch{}".format(self.epoch)
        model.save(saved)
        print(
            "Epoch saved: {}".format(self.epoch + 1),
            "Start next epoch"
        )
        self.epoch += 1
        
# Traning start from here       
def Train(checkpoint=True):
    '''
    Default checkpoint is true.
    Model will be save after every epoch
    '''
    with open("data/bn_corpus.pickle", "rb") as f:
        data = pickle.load(f)
    train_data = [txt.split(" ") for txt in data]

    del data
    
    
    if checkpoint:
        model = Word2Vec(train_data, sg=sg, window=window,size=size,
			min_count=min_count, workers=workers, iter=iters, sample=sample,
			callbacks=[EpochSaver("./checkpoints")])
        model.save("./results/word2vec_new.model")
        print(f"Training Completed. File saved as \" word2vec_new \" in the results folder ")
        
    else:
        model = Word2Vec(train_data, sg=sg, window=window,size=size,
			min_count=min_count, workers=workers, iter=iters, sample=sample)
        model.save("./results/word2vec_new.model")
        print(f"Training Completed. File saved as \" word2vec_new \" in the results folder ")
        
if __name__ == '__main__':       
    Train(checkpoint)       
        
