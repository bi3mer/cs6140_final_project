from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from tqdm import tqdm
import pandas as pd
import os

vids = pd.read_csv('vids.csv')


processed_titles = []
processed_descriptions = []
processed_tags = []

for i in tqdm(range(len(vids))):
    try:
        processed_titles.append(simple_preprocess(vids.title.iloc[i]))
    except:
        processed_titles.append([""])

    try:
        processed_descriptions.append(simple_preprocess(vids.description.iloc[i]))
    except:
        processed_descriptions.append([""])

    try:
        processed_tags.append(simple_preprocess(vids.tags.iloc[i]))
    except:
        processed_tags.append([""])

class Corpus:
    def __iter__(self):
        for title in tqdm(processed_titles):
            yield title
            
        for description in tqdm(processed_descriptions):
            yield description
            
        for tag in tqdm(processed_tags):
            yield tag
           
w2v_model = Word2Vec(sentences=Corpus())

if not os.path.exists('mdl'):
    os.mkdir('mdl')

w2v_model.save(os.path.join('mdl', 'word2vec.model'))