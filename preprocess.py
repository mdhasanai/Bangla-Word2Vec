import os
import json
import pickle


text_file_name = "data/bangla_text.txt"
saved_file_name = "data/bn_corpus.pickle"

'''
This scipt is going to make a list of sentenses from an raw text file.
Finally, it will save the file as pickle format in the data folder as "bn_corpus.pickle"
'''
def preprocess():

    with open(text_file_name, "r") as f:
        text = f.read()
    sentences = text.split("।")

    # There could be \n in the sentences. It should be removed
    sentences = [sen.replace("\n","") for sen in sentences] 
    print(f"\nTotal Bangla sentences: {len(sentences)} ")

    with open(saved_file_name, "wb") as files:
        pickle.dump(sentences, file=files )
    print(f"bn_corpus file saved in the data/ folder ")
    
if __name__ == '__main__':
    preprocess()
