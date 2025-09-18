# image_captioning_train.py

import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.preprocessing.text import Tokenizer
import os
import pickle

def clean_captions(captions):
    table = str.maketrans('', '', string.punctuation)
    cleaned = {}
    for key, desc_list in captions.items():
        cleaned[key] = []
        for line in desc_list:
            line = line.lower()
            line = line.translate(table)
            line = line.split()
            line = [word for word in line if len(word) > 1]
            line = ' '.join(line)
            cleaned[key].append("startseq " + line + " endseq")
    return cleaned

def extract_features(directory):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = {}
    for name in os.listdir(directory):
        filename = os.path.join(directory, name)
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature
    return features

def create_tokenizer(captions):
    lines = []
    for key in captions:
        for desc in captions[key]:
            lines.append(desc)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def data_generator(captions, photos, tokenizer, max_length, vocab_size):
    while True:
        for key, desc_list in captions.items():
            photo = photos[key][0]
            for desc in desc_list:
                seq = tokenizer.texts_to_sequences([desc])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    yield [[photo, in_seq], out_seq]

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

if __name__ == "__main__":
    with open("captions.pkl", "rb") as f:
        captions = pickle.load(f)
    captions = clean_captions(captions)
    features = extract_features("Flicker8k_Dataset")
    pickle.dump(features, open("features.pkl", "wb"))
    tokenizer = create_tokenizer(captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(c.split()) for cap in captions.values() for c in cap)
    pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))
    model = define_model(vocab_size, max_length)
    generator = data_generator(captions, features, tokenizer, max_length, vocab_size)
    steps = sum(len(captions[key]) for key in captions)
    model.fit(generator, epochs=20, steps_per_epoch=steps, verbose=1)
    model.save("caption_model.h5")
    photo = features[list(features.keys())[0]]
    print("Generated Caption:", generate_caption(model, tokenizer, photo, max_length))
