## Bangla-Word2Vec

### Requirements
- Python 3.5 or later
- Gensim

Install requirements packages by the following command
```
pip install -r requirments.txt
``` 


#### Load Pretrained Bangla Word Embedding in Pytorch
If you want to load the pretrained Embedding weights into your project. Follow the following procedure.

1. Run the following script to export the pretrained weights

```python
export_weights.py
```



### You can train the word2vec model with your own Bangla datasets by the following procedure
copy the text file into the "data/" folder

#### Preprocess:
To preprocess, run the following file
```python
python preprocess.py
``


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
- callbacks = True (if you fon't want to save model after every iteration than make it False) 
- save = 'word2vec/' (automatic create this folder if it doesn't exist)

```python
python train.py
```
#### Evaluate:
To Evaluate, run this script.
```python
python eval.py
```
Note: In terminal, Bangla words can't display properly. So, It is better to evaluate in the jupyter notebook. if you don't have jupyter, inatall anaconda and than run "jupyter notebook" in the terminal.
 

