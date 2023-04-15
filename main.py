from __future__ import print_function
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log, sqrt
from sklearn.model_selection import train_test_split
import numpy as np
import re
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

tweet = pd.read_csv("h.csv", encoding="ISO-8859-1")
tweet.rename(columns={"annotation__label__-": "label", "content": "tweet"}, inplace=True)
tweet.drop(['annotation__notes'], axis=1, inplace=True)
tweet.drop(['extras'], axis=1, inplace=True)

# tweet.label.value_counts()

tweet['tweet'] = tweet['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# Replacement - tweet['tweet'] = tweet['tweet'].str.lower()

tweet['tweet'] = tweet['tweet'].str.replace('[^\w\s]', '')
# removes anything which is not letter or digit
tweet['tweet'] = tweet['tweet'].str.replace(r'_', '')
tweet['numerics'] = tweet['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
tweet['tweet'] = tweet['tweet'].str.replace('[\d+]', '')
# removes any digits from text
stop = stopwords.words('english')
tweet['tweet'] = tweet['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
tweet['tweet'] = tweet['tweet'].apply(lambda x: " ".join(x for x in x.split() if len(x) > 2))

freq = pd.Series(' '.join(tweet['tweet']).split()).value_counts()[:10]
freq = list(freq.index)
tweet['tweet'] = tweet['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

bully_words = ' '.join(list(tweet[tweet['label'] == 1]['tweet']))
# bully_wc = WordCloud(width = 512,height = 380).generate(bully_words)
# plt.figure(figsize = (10, 8), facecolor = (0, 0, 0))
# plt.imshow(bully_wc)
# plt.axis('off')
# plt.tight_layout(pad = 0)
# plt.show()
# print(5)

non_bully_words = ' '.join(list(tweet[tweet['label'] == 0]['tweet']))
non_bully_wc = WordCloud(width=512, height=380).generate(non_bully_words)
# plt.figure(figsize = (10, 8), facecolor = 'k')
# plt.imshow(non_bully_wc)
# plt.axis('off')
# plt.tight_layout(pad = 0)
# plt.show()
# print(6)


# In[19]:


# how to define X and y (from the Tweeter data) for use with COUNTVECTORIZER
X = tweet.tweet
y = tweet.label
print(X.shape)
print(y.shape)

# split X and y into training and testing sets


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=10)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print(8)


# In[21]:


def tokenize(tweet):
    words = word_tokenize(tweet)

    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    lemma = WordNetLemmatizer()
    words = [lemma.lemmatize(word) for word in words]

    return words


# In[22]:


# instantiate the vectorizer
# args = {"stem": True, "lemmatize": False}
vect = CountVectorizer(analyzer='word', binary=False, decode_error='strict',
                       encoding='ISO-8859-1', input='content',
                       lowercase=True, max_df=1.0, max_features=None, min_df=1,
                       ngram_range=(1, 2), preprocessor=None, stop_words='english',
                       strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',
                       tokenizer=tokenize, vocabulary=None)

print(9)
# vect =  TfidfVectorizer(analyzer='word', binary=False, decode_error='strict',
# encoding='ISO-8859-1', input='content',
# lowercase=True, max_df=1.0, max_features=None, min_df=1,
# ngram_range=(1, 1), norm='l2', preprocessor=None, smooth_idf=True,
# stop_words='english', strip_accents=None, sublinear_tf=False,
# token_pattern='(?u)\\b\\w\\w+\\b', tokenizer=tokenize, use_idf=True,
# vocabulary=None)
# vect = HashingVectorizer()


# In[23]:


# equivalently: combine fit and transform into a single step
X_train_dtm = vect.fit_transform(X_train)
print(10)

# In[24]:


# examine the document-term matrix


# In[25]:


# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
print(X_test_dtm)
print(11)

# In[26]:


# import and instantiate a Multinomial Naive Bayes model

nb = MultinomialNB()

nb.fit(X_train_dtm, y_train)

# In[27]:


# train the model using X_train_dtm (timing it with an IPython "magic command")
# get_ipython().run_line_magic('time', 'nb.fit(X_train_dtm, y_train)')


# In[28]:


# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)
accuracy = accuracy_score(y_test, y_pred_class)
print("Accuracy of Naive Bayes model:", accuracy)
print(12)

# In[29]:


# calculate accuracy of class predictions
from sklearn import metrics

metrics.accuracy_score(y_test, y_pred_class)

# In[30]:


# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)
i = 13
print(13)
i = i + 1

# In[31]:


# print the classification report
print(metrics.classification_report(y_test, y_pred_class))

# In[32]:


# calculate predicted probabilities for X_test_dtm (poorly calibrated)
y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]

# In[33]:


# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)

# In[34]:


# import and instantiate a logistic regression model
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
print(14)
i = i + 1

# In[35]:


# train the model using X_train_dtm
# get_ipython().run_line_magic('time', 'lr.fit(X_train_dtm, y_train)')
lr.fit(X_train_dtm, y_train)

# In[36]:


# make class predictions for X_test_dtm
y_pred_class = lr.predict(X_test_dtm)

accuracy = accuracy_score(y_test, y_pred_class)
print("Accuracy of LR:", accuracy)
print(14.1)

# In[37]:


# calculate predicted probabilities for X_test_dtm (well calibrated)
y_pred_prob = lr.predict_proba(X_test_dtm)[:, 1]
print(15)
i = i + 1

# In[38]:


# calculate accuracy
metrics.accuracy_score(y_test, y_pred_class)

# In[39]:


metrics.confusion_matrix(y_test, y_pred_class)

# In[40]:


# print the classification report
print(metrics.classification_report(y_test, y_pred_class))

# In[41]:
print(16)
i = i + 1

# calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)

# In[42]:


from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier(weights='distance', n_neighbors=2)
# get_ipython().run_line_magic('time', 'kn.fit(X_train_dtm, y_train)')
kn.fit(X_train_dtm, y_train)

# In[43]:


y_pred_class = kn.predict(X_test_dtm)
accuracy = accuracy_score(y_test, y_pred_class)
print("Accuracy of KN classifer:", accuracy)

print(17)
i = i + 1

# In[44]:


metrics.accuracy_score(y_test, y_pred_class)

# In[45]:


metrics.confusion_matrix(y_test, y_pred_class)

# In[46]:
print(18)
i = i + 1

# print the classification report
print(metrics.classification_report(y_test, y_pred_class))

# In[47]:


metrics.roc_auc_score(y_test, y_pred_prob)

# In[48]:


from sklearn import tree

dc = tree.DecisionTreeClassifier(criterion='entropy')
# get_ipython().run_line_magic('time', 'dc.fit(X_train_dtm, y_train)')
dc.fit(X_train_dtm, y_train)

# In[49]:


y_pred_class = dc.predict(X_test_dtm)
accuracy = accuracy_score(y_test, y_pred_class)
print("Accuracy of DTC:", accuracy)

print(19)
i = i + 1
# In[50]:


metrics.accuracy_score(y_test, y_pred_class)

# In[51]:


metrics.confusion_matrix(y_test, y_pred_class)

# In[52]:


# print the classification report
print(metrics.classification_report(y_test, y_pred_class))

# In[53]:


metrics.roc_auc_score(y_test, y_pred_prob)

# In[54]:


from sklearn import svm

svl = svm.LinearSVC()
# get_ipython().run_line_magic('time', 'svl.fit(X_train_dtm, y_train)')
svl.fit(X_train_dtm, y_train)

# In[55]:


y_pred_class = svl.predict(X_test_dtm)
accuracy = accuracy_score(y_test, y_pred_class)
print("Accuracy of svc:", accuracy)
print(20)
i = i + 1

# In[56]:


metrics.accuracy_score(y_test, y_pred_class)

# In[57]:


metrics.confusion_matrix(y_test, y_pred_class)

# In[58]:


# print the classification report
print(metrics.classification_report(y_test, y_pred_class))

# In[59]:


metrics.roc_auc_score(y_test, y_pred_prob)
print(21)
i = i + 1

# In[60]:


sv = svm.SVC(gamma=10)
# get_ipython().run_line_magic('time', 'sv.fit(X_train_dtm, y_train)')
sv.fit(X_train_dtm, y_train)

# In[61]:


y_pred_class = sv.predict(X_test_dtm)

accuracy = accuracy_score(y_test, y_pred_class)
print("Accuracy of svc:", accuracy)

# In[62]:


metrics.accuracy_score(y_test, y_pred_class)

# In[63]:


metrics.confusion_matrix(y_test, y_pred_class)
print(22)
i = i + 1

# In[64]:


# print the classification report
print(metrics.classification_report(y_test, y_pred_class))

# ## Examining a model for further insight
#

# In[65]:


# store the vocabulary of X_train
X_train_tokens = vect.get_feature_names_out()
len(X_train_tokens)

# In[66]:
print(23)
i = i + 1

# first 10 false positives (Nonbullying materials classified as bullying)
X_test[y_test < y_pred_class].head(10)

# In[67]:


# first 10 false negatives (bullying materials classified as nonbullying)
X_test[y_test > y_pred_class].head(10)

# In[68]:


# examine the first 100 tokens
print(X_train_tokens[0:100])

# In[69]:


# examine the last 100 tokens
print(X_train_tokens[-100:])

# In[70]:


# Naive Bayes counts the number of times each token appears in each class
nb.feature_count_

# In[71]:
print(24)
i = i + 1

# rows represent classes, columns represent tokens
nb.feature_count_.shape

# In[72]:


# number of times each token appears across all Nonbullying messages
nonbullying_token_count = nb.feature_count_[0, :]
nonbullying_token_count

# In[73]:


# number of times each token appears across all Bullying messages
bullying_token_count = nb.feature_count_[1, :]
bullying_token_count

# In[74]:


# create a DataFrame of tokens with their separate counts
tokens = pd.DataFrame(
    {'token': X_train_tokens, 'Nonbullying': nonbullying_token_count, 'Bullying': bullying_token_count})
tokens.head(10)

# In[75]:


# examine 10 random DataFrame rows
tokens.sample(10, random_state=100)

# In[76]:
print(25)
i = i + 1

# Naive Bayes counts the number of observations in each class
nb.class_count_

# In[77]:


# add 1 to counts to avoid dividing by 0
tokens['Nonbullying'] = tokens.Nonbullying + 1
tokens['Bullying'] = tokens.Bullying + 1
tokens.sample(10, random_state=100)

# In[78]:


# convert the bullying and non bullying counts into frequencies
tokens['Nonbullying'] = tokens.Nonbullying / nb.class_count_[0]
tokens['Bullying'] = tokens.Bullying / nb.class_count_[1]
tokens.sample(10, random_state=100)

# In[79]:
print(26)
i = i + 1

# calculate the ratio of bullying for each token
tokens['Bullying_ratio'] = tokens.Bullying / tokens.Nonbullying
random_tokens = tokens.sample(10, random_state=100)

# In[80]:


# examine the DataFrame sorted by spam_ratio
# note: use sort() instead of sort_values() for pandas 0.16.2 and earlier
# tokens = tokens.sort_values('Bullying_ratio', ascending=False)
fig, ax = plt.subplots()
ax.barh(random_tokens['token'], random_tokens['Bullying_ratio'])
ax.set(xlabel='Bullying ratio', ylabel='Tokens')

# In[81]:


sample_test = ["You are not an idiot", "You dirty piece of shit", "I want to save you from being killed",
               "Threaten him to death", "try to help him", "show your real face",
               "I think we should appreciate that guy", "Join us so that we can enjoy more", "I think he likes you",
               "I do not believe you", "Help me understand this concept",
               "Don't you think he is the best for this job?", "Let us try to fix this situation",
               "I think he did really great", "Make sure you study that", "Grab and scratch his hand",
               "I cannot believe you did it", "Please call him by his first name", "Enhance the audio",
               "Please help me", "Cater to the needs of the society"]

# In[82]:


sample_test_dtm = vect.transform(sample_test)
print(sample_test_dtm)

# In[83]:


print(X_train_tokens[37531])
print(27)
i = i + 1

# In[84]:


sample_pred_nb = nb.predict(sample_test_dtm)
sample_pred_lr = lr.predict(sample_test_dtm)
sample_pred_kn = kn.predict(sample_test_dtm)
sample_pred_dc = dc.predict(sample_test_dtm)
sample_pred_svl = svl.predict(sample_test_dtm)
sample_pred_sv = sv.predict(sample_test_dtm)

# In[85]:


print("naive bayes predicted:           ", sample_pred_nb)
print("logistic regression predicted:   ", sample_pred_nb)
print("k-nearest neighbours predicted:  ", sample_pred_nb)
print("decision tree predicted:         ", sample_pred_nb)
print("linear support vector predicted: ", sample_pred_nb)
print("rbf support vector predicted:    ", sample_pred_nb)

# In[ ]:

print(28)
i = i + 1


