# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#
#  This is a pythin file that takes a deep look into the dataset provided by kaggle
#  I look at target distribution and question length distribution. The mian
#  purpose of this was to develop figures for our poster and project report.
#
#

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_text = train_data['question_text']
test_text = test_data['question_text']
frames = [train_text, test_text]
all_text = pd.concat(frames, sort=False)
print("train shape: " + str(train_data.shape))
print("test shape: " + str(test_data.shape))
train_data.head()

print("train size: " + str(len(train_data)))
print("test size: " + str(len(test_data)))
print("All data size: " + str(len(all_text)))

train_data.groupby('target').size()

num_zeros = len(train_data[train_data["target"]==0])
num_ones = len(train_data[train_data["target"]==1])
insin_percentage = (num_ones) / (num_ones+num_zeros)
print("Insincere ratio: " + str(insin_percentage))

fig, ax = plt.subplots()
n, bins, patches = ax.hist(train_data['target'], 2)
ax.set_xlabel('Label')
ax.set_ylabel('Label Frequency')
ax.set_title(r'Distribution of targets in training data')
fig.tight_layout()
plt.show()

labels = 'Sincere', 'Incinsere'
sizes = [(1-insin_percentage), insin_percentage]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.show()

q_length = []
for index, row in all_text.iteritems():
    q_length.append(len(row.split()))

min_len = min(q_length)
max_len = max(q_length)
avg_len = sum(q_length) / len(q_length)
print("Shortest question length: " + str(min_len))
print("Longest question length: " + str(max_len))
print("Average quesiton length: " + str(avg_len))
np.percentile(q_length, 99.9999)
print(str(np.percentile(q_length, 99.9999)))

fig, ax = plt.subplots()
n, bins, patches = ax.hist(q_length, 100)
ax.set_xlabel('Question Length')
ax.set_ylabel('Frequency')
ax.set_title(r'Distribution of question length')
fig.tight_layout()
plt.show()

sns.distplot(q_length, bins=100, kde=False, rug=True);