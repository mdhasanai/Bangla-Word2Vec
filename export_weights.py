import os
import gensim
from gensim import corpora
from pprint import pprint
import pickle
from gensim.models import Word2Vec


def export_weight():
    '''
    Model wight will be saved in results folder
    '''
	model_path = "results/word2vec.model"
	model = Word2Vec.load(model_path)
	with open("results/pretrained_weights.pickle", "wb") as file:
		pickle.dump(model.wv.syn0,file)

    print(f"Pretained wights successfully saved.")

# exporting wights 		
if __name__ == '__main__':
     
    export_weight()  
