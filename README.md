## Bangla-Word2Vec

What is Word2vec?

Word2vec is a method to efficiently create word embeddings. Word2vec model is basically a two-layer neural network that processes text. The use of word2vec is huge in deep learning such as Machine translations, Language modelling, Question and Answering, Image Captioning, Speech Recognition and so on. You can use this bangla word2vec model in your projects for getting better result. The procedure is given below to load this pretained weight into the project.


### Requirements
- Python > 3.5
- Gensim

Install requirements packages by the following command
```python
pip install -r requirments.txt
```


#### Load Pretrained Bangla Word Embedding in Pytorch
If you want to load the pretrained Embedding weights into your project. Follow the following procedure.

1. Run the following script to export the pretrained weights

```python
python export_weights.py
```
it will export the pretrained weight as "pretrained_weights.pickle" in the results folder.

2. Load the pretrained weights

```python
import pickle
import torch.nn 

embedding = nn.Embedding(num_embeddings, embedding_dim=300)
with open("results/pretrained_weights.pickle","rb") as f:
    weight = pickle.load(f)
weight = torch.from_numpy(weight)
embedding.weight = nn.Parameter(weight)

```



### You can train the word2vec model with your own Bangla datasets by the following procedure
copy the text file into the "data/" folder

#### Preprocess:
To preprocess, run the following script
```python
python preprocess.py
```


#### Train:
To train the model, run the following script.
Default configuration for training,
- sg = 1 (skip-gram. For training with CBOW, make it 0)
- window = 4
- size = 300 (vector dimension)
- min_count = 2
- workers = 8
- iter    = 100
- sample  = 0.01
- callbacks = True (if you don't want to save the model after every iteration than make it False) 
- save = 'results/' (automatic create this folder if it doesn't exist)

```python
python train.py
```
#### Evaluate:
To Evaluate, run this script.
```python
python eval.py
```
Note: In terminal, Bangla words can't be displayed properly. So, It is better to evaluate them in the jupyter notebook. if you don't have jupyter, install anaconda python properly and run "jupyter notebook" in the terminal.



#### Feel free to create an issue if you (the reader) come across any problems. 




