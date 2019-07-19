## Bangla-Word2Vec

### Requirements
- Python 3.5 or later
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

embedding = nn.Embedding(num_embeddings, embedding_dim=300 )
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
- sp = 1 (skip-gram)
- window = 4
- size = 300 (vector dimension)
- min_count = 2
- workers = 8
- iter    = 100
- sample  = 0.01
- callbacks = True (if you don't want to save model after every iteration than make it False) 
- save = 'results/' (automatic create this folder if it doesn't exist)

```python
python train.py
```
#### Evaluate:
To Evaluate, run this script.
```python
python eval.py
```
Note: In terminal, Bangla words can't display properly. So, It is better to evaluate in the jupyter notebook. if you don't have jupyter, inatall anaconda and than run "jupyter notebook" in the terminal.
 

