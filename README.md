# cs6140_final_project

## Data Pre-Processing

YouTUbe data can be downloaded [here](https://www.kaggle.com/datasnaek/youtube-new/download). Thumbnail images can be downloaded [here](https://www.kaggle.com/kamalhaddad/thumbnail-images/download). Use [config.py](./config.py) to configure the path to each. I changed the folder name from 'archive' to 'youtube' and 'img' to 'youtube-thumbnail' but you can do whatever you want.

To build `vids.csv` you can either run `python build_data.py` or download it [here](https://www.kaggle.com/kamalhaddad/vidswithencoding/download). Make sure that you take the CSV and put it into this project's directory.

## Word2Vec

Run `python build_w2v_model.py`. This creates the `mdl` directory and `word2vec.model` which we will use to encode strings as input.