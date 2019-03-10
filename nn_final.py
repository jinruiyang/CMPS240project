# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import re
print(os.listdir("../input"))

import time

import tensorflow as tf
from tensorflow import keras

from sklearn import metrics
from sklearn.model_selection import train_test_split

from keras.preprocessing import text, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Concatenate, concatenate, Activation, CuDNNGRU, CuDNNLSTM, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

# Any results you write to the current directory are saved as output.

mispell_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
                "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "i'd": "i would",
                "i'd've": "i would have", "i'll": "i will", "i'll've": "I will have", "i'm": "i am",
                "i've": "I have", "isn't": "is not", "it'd": "it would",
                "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have",
                "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
                "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
                "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not",
                "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
                "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
                "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                "there'd've": "there would have", "there's": "there is", "here's": "here is",
                "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                "they'll've": "they will have", "they're": "they are", "they've": "they have",
                "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have",
                "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
                "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color',
                'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling',
                'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor',
                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize',
                'youtu ': 'youtube ', 'qoura': 'quora', 'sallary': 'salary', 'whta': 'what',
                'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can',
                'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doi': 'do I',
                'thebest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation',
                'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis',
                'etherium': 'ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017',
                '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess',
                "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization',
                'demonitization': 'demonetization', 'demonetisation': 'demonetization'}

puncts = '\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2",
                 "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '”': '"', '“': '"', "£": "e",
                 '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta',
                 '∅': '', '³': '3', 'π': 'pi', '\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}
for p in puncts:
    punct_mapping[p] = ' %s ' % p

p = re.compile('(\[ math \]).+(\[ / math \])')
p_space = re.compile(r'[^\x20-\x7e]')

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

def clean_text(text):
    # clean latex maths
    text = p.sub(' [ math ] ', text)
    # clean invisible chars
    text = p_space.sub(r'', text)
    # clean punctuations
    for punct in punct_mapping:
        if punct in text:
            text = text.replace(punct, punct_mapping[punct])
    tokens = []
    for token in text.split():
        # replace contractions & correct misspells
        token = mispell_dict.get(token.lower(), token)
        tokens.append(token)
    text = ' '.join(tokens)
    return text


# extracting features #
print("Extracting features......")
train_text = train_data['question_text'].apply(clean_text)
test_text = test_data['question_text'].apply(clean_text)
train_target = train_data['target']
all_text = train_text.append(test_text)

# define some values for Neural Network model #
max_num_words = 50000       # number of words to keep as features
max_len = 72                # length of every instance
embed_size = 300            # size of each word_vector

# creating the tokenizer #
print("Creating the tokenizer......")
token = text.Tokenizer(num_words=max_num_words)
token.fit_on_texts(all_text)
word_index = token.word_index

# convert train_text to vectors of ints #
print("Converting text to sequences......")
train_seq_texts = token.texts_to_sequences(train_text)
test_seq_texts = token.texts_to_sequences(test_text)

# pad sentences so they're all the same lnegth #
print("Padding sequences to all the same length......")
train_seq_x = sequence.pad_sequences(train_seq_texts, maxlen=max_len)
test_seq_x = sequence.pad_sequences(test_seq_texts, maxlen=max_len)

# Split train_text into train and validation sets: 90-10 split #
print("Getting train and validation sets......")
train_x, valid_x, train_y, valid_y = train_test_split(train_seq_x, train_target, test_size=.1)
print ("Length of training set: " + str(len(train_x)))
print ("Length of validation set: " + str(len(valid_x)))


###################create wiki embedidngs matrix############################
EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

nb_words = min(max_num_words, len(word_index))
wiki_embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_num_words: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: wiki_embedding_matrix[i] = embedding_vector


###################create glove embedidngs matrix############################
EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

nb_words = min(max_num_words, len(word_index))
#nb_words = len(word_index) - 1
glove_embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_num_words: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: glove_embedding_matrix[i] = embedding_vector


###################create paragram embedidngs matrix############################
EMBEDDING_FILE = '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o)>100)

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

nb_words = min(max_num_words, len(word_index))
paragram_embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_num_words: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: paragram_embedding_matrix[i] = embedding_vector


def create_model(embedding_matrix):
    inp = Input(shape=(max_len,))
    if embedding_matrix is None:
        x = Embedding(max_num_words, embed_size)(inp)
    else:
        x = Embedding(max_num_words, embed_size, weights=[embedding_matrix])(inp)

    x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)

    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)

    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


############ train model with wiki embeddings #################
model = create_model(wiki_embedding_matrix)
print(model.summary())

model.fit(train_x, train_y, batch_size=512, epochs=2, validation_data=(valid_x, valid_y))

wiki_pred_valid_y = model.predict([valid_x], batch_size=512, verbose=1)

best_score = 0.0
wiki_best_thresh = 0.0
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    score = metrics.f1_score(valid_y, (wiki_pred_valid_y>thresh).astype(int))
    if score > best_score:
        wiki_best_thresh = thresh
        best_score = score
print("Best threshold: " + str(wiki_best_thresh))
print("Val F1 Score: {:.6f}".format(best_score))

wiki_pred_test_y = model.predict([test_seq_x], batch_size=1024, verbose=1)
del model

############# train model with paragram embeddings #################
model = create_model(paragram_embedding_matrix)
print(model.summary())

model.fit(train_x, train_y, batch_size=512, epochs=2, validation_data=(valid_x, valid_y))

para_pred_valid_y = model.predict([valid_x], batch_size=512, verbose=1)

best_score = 0.0
para_best_thresh = 0.0
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    score = metrics.f1_score(valid_y, (para_pred_valid_y>thresh).astype(int))
    if score > best_score:
        para_best_thresh = thresh
        best_score = score
print("Best threshold: " + str(para_best_thresh))
print("Val F1 Score: {:.6f}".format(best_score))

para_pred_test_y = model.predict([test_seq_x], batch_size=1024, verbose=1)
del model



############### train model with glove embeddings ####################
model = create_model(glove_embedding_matrix)
print(model.summary())

model.fit(train_x, train_y, batch_size=512, epochs=2, validation_data=(valid_x, valid_y))

glove_pred_valid_y = model.predict([valid_x], batch_size=512, verbose=1)

best_score = 0.0
glove_best_thresh = 0.0
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    score = metrics.f1_score(valid_y, (glove_pred_valid_y>thresh).astype(int))
    if score > best_score:
        glove_best_thresh = thresh
        best_score = score
print("Best threshold: " + str(glove_best_thresh))
print("Val F1 Score: {:.6f}".format(best_score))

glove_pred_test_y = model.predict([test_seq_x], batch_size=1024, verbose=1)
del model


################### test model with different variations ################
w_coe = .3
p_coe = .2
g_coe = .5
pred_val_y = w_coe*wiki_pred_valid_y + p_coe*para_pred_valid_y + g_coe*glove_pred_valid_y

best_score = 0.0
best_thresh = 0.0
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    #print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(valid_y, (pred_val_y>thresh).astype(int))))
    score = metrics.f1_score(valid_y, (pred_val_y>thresh).astype(int))
    if score > best_score:
        best_thresh = thresh
        best_score = score
print("Best threshold: " + str(best_thresh))
print("Val F1 Score: {:.6f}".format(best_score))


####### create csv ############
prediction = w_coe*wiki_pred_test_y + p_coe*para_pred_test_y + g_coe*glove_pred_test_y
final_prediction = (prediction>best_thresh).astype(int)
out_df = pd.DataFrame({"qid":test_data["qid"].values})
out_df['prediction'] = final_prediction
out_df.to_csv("submission.csv", index=False)