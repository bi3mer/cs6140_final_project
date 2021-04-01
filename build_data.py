from keras.preprocessing.image import ImageDataGenerator
from keras import layers
import tensorflow as tf
import keras

from PIL import Image
import pandas as pd
import numpy as np
import glob
import cv2
import os

import config


vids = pd.read_csv(os.path.join(config.YOUTUBE_DATA_PATH, 'USvideos.csv'))
vid_cats = pd.read_json(os.path.join(config.YOUTUBE_DATA_PATH, 'US_category_id.json'))

# category id to string
cat_id_to_name = {}
for cat in vid_cats['items']:
    cat_id_to_name[cat['id']] = cat['snippet']['title']
vids.insert(len(vids.columns), 'category', vids['category_id'].astype(str).map(cat_id_to_name))

# CNN Encoding
img_path = os.path.join(config.THUMBNAIL_DATA_PATH, '*')
images = []
video_ids = []

for f in glob.iglob(img_path):
    video_ids.append(os.path.basename(f).split('.')[0])
    img_array = np.asarray(Image.open(f))
    images.append(cv2.normalize(img_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

images = np.array(images)

class CNNAutoEncoder():
    def __init__(self):
        self.img_rows = 90
        self.img_cols = 120
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
#         optimizer = tf.keras.optimizers.Adam(lr=0.002)
        
        self.autoencoder_model = self.build_model()
        self.autoencoder_model.compile(loss='binary_crossentropy', optimizer='adam')
        self.autoencoder_model.summary()
    
    def build_model(self):
        input_img = keras.Input(shape=self.img_shape)
        encoded = layers.Flatten()(input_img)
        
#         encoded = layers.Dense(1000, activation='relu')(encoded)
        encoded = layers.Dense(100, activation='relu')(encoded)
        encoded = layers.Dense(5, activation='relu', name='encoding_layer')(encoded)
    
        decoded = layers.Dense(100, activation='relu')(encoded)
#         decoded = layers.Dense(1000, activation='relu')(encoded)
        decoded = layers.Dense(32400, activation='sigmoid')(decoded)
        decoded = layers.Reshape((self.img_rows, self.img_cols, self.channels))(decoded)
        autoencoder = keras.Model(input_img, decoded)
#         autoencoder.compile(optimizer='adam', loss='mse')
        return keras.Model(input_img, decoded)
    
    def train_model(self, x_train, y_train, epochs=30, batch_size=32):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=5,
                                       verbose=1, 
                                       mode='auto')

        history = self.autoencoder_model.fit(x_train, y_train,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_split=0.2,
                                             callbacks=[early_stopping])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    
    def evaluate(self, x_test):
        pred = self.autoencoder_model.predict(x_test)
        return pred
    
    def get_encoding(self, x_test):
        intermediate_model = keras.Model(self.autoencoder_model.input, self.autoencoder_model.get_layer('encoding_layer').output)
        return intermediate_model.predict(x_test)

cnn_ae = CNNAutoEncoder()
cnn_ae.train_model(images, images)

# store encodings
encodings = cnn_ae.get_encoding(images)
df = pd.DataFrame(video_ids, columns=['video_id'])
img_feature_names = []
for i in range(encodings.shape[1]):
    img_feature_names.append(f"img_feature_{i}")
df[img_feature_names] = pd.DataFrame(encodings)

# save result
vids.to_csv('vids.csv', index=False)