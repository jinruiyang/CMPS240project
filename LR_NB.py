# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import re
print(os.listdir("input"))
import nltk
from nltk.corpus import stopwords

from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,ComplementNB,BernoulliNB

# added for Neural Network #
import time

import tensorflow as tf
from tensorflow import keras

from sklearn import metrics
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split



# Any results you write to the current directory are saved as output.

train_path = 'input/train.csv'
test_path = 'input/test.csv'

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

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

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

print(train_data.shape)

print(train_data.head())

print(train_data['target'].value_counts())

##########extract features#############

print('extracting features.......')
train_text = train_data['question_text'].apply(clean_text)
test_text = test_data['question_text'].apply(clean_text)
train_target = train_data['target']
all_text = train_text.append(test_text)

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(all_text)

count_vectorizer = CountVectorizer()
count_vectorizer.fit(all_text)

train_text_features_cv = count_vectorizer.transform(train_text)
test_text_features_cv = count_vectorizer.transform(test_text)

train_text_features_tf = tfidf_vectorizer.transform(train_text)
test_text_features_tf = tfidf_vectorizer.transform(test_text)

##########Logistic regression#############
print('Logistic regression predicting......')
kfold = KFold(n_splits = 5, shuffle = True, random_state = 2018)
test_preds = 0
oof_preds = np.zeros([train_data.shape[0],])

for i, (train_idx,valid_idx) in enumerate(kfold.split(train_data)):
    x_train, x_valid = train_text_features_tf[train_idx,:], train_text_features_tf[valid_idx,:]
    y_train, y_valid = train_target[train_idx], train_target[valid_idx]
    classifier = LogisticRegression()
    #print('fitting.......')
    classifier.fit(x_train,y_train)
    #print('predicting......')
    #print('\n')
    oof_preds[valid_idx] = classifier.predict_proba(x_valid)[:,1]
    test_preds += 0.2*classifier.predict_proba(test_text_features_tf)[:,1]

pred_train = (oof_preds > .25).astype(np.int)
lr_acc = f1_score(train_target, pred_train)
print("Logistic Regression model accuracy is {:.2f}%" .format(lr_acc * 100))

submission1 = pd.DataFrame.from_dict({'qid': test_data['qid']})
submission1['prediction'] = (test_preds>0.25).astype(np.int)
# submission1.to_csv('logistic_submission.csv', index=False)
submission1['prediction'] = (test_preds>0.25)

##############Naive Bayes############
print('Naive Bayes predicting......')
kfold = KFold(n_splits = 5, shuffle = True, random_state = 2018)
test_preds1 = 0
oof_preds1 = np.zeros([train_data.shape[0],])

test_preds2 = 0
oof_preds2 = np.zeros([train_data.shape[0],])

for i, (train_idx,valid_idx) in enumerate(kfold.split(train_data)):
    x_train, x_valid = train_text_features_cv[train_idx,:], train_text_features_cv[valid_idx,:]
    y_train, y_valid = train_target[train_idx], train_target[valid_idx]
    classifier1 = MultinomialNB()
    classifier2 = BernoulliNB()
    #print('fitting.......')
    classifier1.fit(x_train,y_train)
    classifier2.fit(x_train,y_train)
    #print('predicting......')
    #print('\n')
    oof_preds1[valid_idx] = classifier1.predict_proba(x_valid)[:,1]
    test_preds1 += 0.2*classifier1.predict_proba(test_text_features_cv)[:,1]
    oof_preds2[valid_idx] = classifier2.predict_proba(x_valid)[:,1]
    test_preds2 += 0.2*classifier2.predict_proba(test_text_features_cv)[:,1]

pred_train = (oof_preds1 > .3).astype(np.int)
MultinomialNB_acc = f1_score(train_target, pred_train)
print("Multinomial Naive Bayes model accuracy is {:.2f}%" .format(MultinomialNB_acc * 100))

pred_train = (oof_preds2 > .3).astype(np.int)
BernoulliNB_acc = f1_score(train_target, pred_train)
print("Bernoulli Naive Bayes model accuracy is {:.2f}%" .format(BernoulliNB_acc * 100))

submission2 = pd.DataFrame.from_dict({'qid': test_data['qid']})
submission3 = pd.DataFrame.from_dict({'qid': test_data['qid']})

submission2['prediction'] = (test_preds1>0.3).astype(np.int)
# submission2.to_csv('multinomial_submission.csv', index=False)
submission2['prediction'] = (test_preds1>0.3)

submission3['prediction'] = (test_preds2>0.3).astype(np.int)
# submission3.to_csv('bernoulli_submission.csv', index=False)
submission3['prediction'] = (test_preds2>0.3)


################ Baseline NN model with no embeddings #################

# defining some values for NN #
max_num_words = 50000
max_len = 70
embed_size = 300

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

# Build NN model #
print("Building model......")
model = keras.Sequential()
model.add(keras.layers.Embedding(max_num_words, embed_size))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Fit NN model #
print("Fitting model......")
history = model.fit(train_x,
                    train_y,
                    epochs=2,
                    batch_size=512,
                    validation_data=(valid_x, valid_y),
                    verbose=1)

# Test model at different thresholds #
print("Testing at different thresholds......")
pred_valid_y = model.predict([valid_x], batch_size=1024, verbose=1)
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    print("F1 score at threshold {0} is {1}".format(thresh, metrics.f1_score(valid_y, (pred_valid_y>thresh).astype(int))))

##########################################################################
submission_final = pd.DataFrame.from_dict({'qid':test_data['qid']})
submission_final['prediction'] = ((0.6*submission1['prediction'] + 0.2*submission2['prediction'] + 0.2*submission3['prediction'])>0.4).astype(np.int)
submission_final.to_csv('submission.csv',index = False)