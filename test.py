import numpy as np
import csv
import nltk
import re
from string import punctuation


#File path for training data
TRAIN = "../quora_data/train.csv"


# This is a list of all stop words to be removed during the pre-processing phase. These words were chosen
# by looking at the most frequent words from our dictionary and removing the ones with no value.
# For now I haven't removed any words that form the start of a question. e.g. how, what , when

stop_words = ['the', 'is',  'a', 'to', 'in', 'of', 'i', 'how', 'do', 'and', 'are', 'for', 'you', 'it',
			  'my', 'if', 'with', 'not', 'on', 'or', 'be', 'from', 'an', 'get', 'as', 'at', 'by', 'we',
			  'so', 'any', 'me', 'am', 'one', 'but', 'all', 'use', 'way', 'up']

# Remove stop words from a str
def remove_stop_words(str):
	tokens = nltk.word_tokenize(str)
	filtered = [w for w in tokens if not w in stop_words]
	str = ' '.join(filtered)
	return str


# The contradictions dictionrary was taken from the wikipedia page for common english contractions. I left
# some out that I didn't think would be necessary, will add/remove more later if need be.

contractions_dict = {
	"ain't": "is not", "amn't": "am not", "aren't": "are not", "can't": "cannot", "could've": "could have",
	"couldn't": "could not", "couldn't've": "could not have", "didn't": "did not", "doesn't": "does not",
	"don't": "do not", "everyone's": "everyone is", "finna": "going to", "gimme": "give me", "gonna": "going to",
	"gotta": "got to", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": " he would", 
	"he's": "he is", "he've": "he have", "how'd": "how did", "how'll": "how will", "how're": "how are", 
	"it'd": "it would", "it'll": "it will", "i'm" : "i am", "it's" : "it is", "let's": "let us",  "ma'am": "madam", "mayn't": "may not",
	"might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
	"mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
	"o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
	"sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
	"she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have",
	"shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have", "so's": "so is", 
	"that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there had",
	"there'd've": "there would have", "there's": "there is", "they'd": "they would", "they'd've": "they would have",
	"they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
	"wasn't": "was not", "we'd": "we had", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
	"we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
	"what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
	"where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
	"who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
	"won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
	"y'all": "you all", "y'alls": "you alls", "y'all'd": "you all would", "y'all'd've": "you all would have", 
	"y'all're": "you all are", "y'all've": "you all have", "you'd": "you had", "you'd've": "you would have", 
	"you'll": "you you will", "you'll've": "you you will have", "you're": "you are", "you've": "you have"
}

# I got this code from stak overflow, it simply looks for a contraction in the contraction dictionary and replaces
# it with the spelled out version. 

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))
def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

# Simple helper fucntion that removes all punctuation from a string
def remove_punctuation(str):
	return ''.join(c for c in str if c not in punctuation)



# OPEN A CSV FILE
# This function will get the data from the csv file
# and create a list of instances where each instance
# contains the question id, the text of the question
# and the label. 
# Parameters: csv file
def getData(filename):
	data = []

	with open(filename) as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=",")
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				line_count += 1
			else:
                                instance = createInstance(row)
				data.append(instance)
				line_count += 1
	return data


# Create a dictionary structure containing all the words 
# in the data structure. This function utilizes NLTK"s
# tokenize function to slpit words in the sentence. 
def createDict(data):
	dictionary = {}
	for elem in data:
		txt = str(elem[1])
		tokens = nltk.word_tokenize(txt)
		addtoDict(dictionary, tokens)
	return dictionary

# This is a helper function for createDict() that simply
# updates the frequency of a word in the dictionary, or
# creates a new entry if the word is not in the dictionary 
# yet.
def addtoDict(dictionary, tokens):
	for word in tokens:
		if word in dictionary:
			dictionary[word] += 1
		else:
			dictionary[word] = 1


# This function creates an instance with the question_id
# It returns the instance and also the txt of the instance
# text and label of a given data entry. It also calls
# pre-process, which does some preprocessing on the txt
def createInstance(row):
	instance = []
	q_id = row[0]
	txt = pre_process(str(row[1]))
	#txt = str(row[1])
	label = row[2]
	instance.append(q_id)
	instance.append(txt)
	instance.append(label)
	return instance

# Preprocess a given string. It does the following in order:
# Converts all strings to lowercase
# Expands all known contractions
# Removes all punctuation
def pre_process(str):
	str = str.lower()
	str = expand_contractions(str)
	str = remove_punctuation(str)
	str = remove_stop_words(str)
	return str	

# Debug function that prints a random number of instances to the
# console.
def randData(elements, data):
	sample = np.random.uniform(low=0, high=len(data), size=elements)
	for num in sample:
		x = int(num)
		print(data[x][1])


def main():
	data = getData(TRAIN)
	dictionary = createDict(data)
	print("Dictionary length: "+ str(len(dictionary)))
	count = 0
	for word in sorted(dictionary, key = dictionary.get, reverse=True):
		if count < 300:
			print (word, dictionary[word])
			count += 1
		else:
			break
	#randData(100, data)

main()
