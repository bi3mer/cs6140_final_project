from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
from tqdm import tqdm

processed_titles = []
processed_descriptions = []
processed_tags = []

for i in tqdm(range(len(X))):
    try:
        processed_titles.append(simple_preprocess(X.title.iloc[i]))
    except:
        processed_titles.append([""])

    try:
        processed_descriptions.append(simple_preprocess(X.description.iloc[i]))
    except:
        processed_descriptions.append([""])

    try:
        processed_tags.append(simple_preprocess(X.tags.iloc[i]))
    except:
        processed_tags.append([""])

class Corpus:
    def __iter__(self):
        print('iter called!')
        for title in processed_titles:
            yield title
            
        for description in processed_descriptions:
            yield description
            
        for tag in processed_tags:
            yield tag
           
w2v_model = Word2Vec(sentences=Corpus())