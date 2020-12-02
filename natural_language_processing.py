# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# why we use tsv file bcz in this two columns value serated by tab
# where in csv it seprated by comma but when in the text it can be a comma in text value so it genrate a problem
# so open tsv file in pandas we need to tell that there is delimiter is tab so in that we need prameter of delimiter='\t'
# so double quoate prblem we need to ignore double quoate so we need to tell pandas to ignore quoate so par. quoating=3
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    # without steeming
    review = review.split()
    # review = [word for word in review if not word in set(stopwords.words('english'))]
    # last 2 lines of output(review)
    # whole experience underwhelming think go ninja sushi next time
    # wasted enough life poured salt wound drawing time took bring check  

    # # with steeming
    ps = PorterStemmer()
    # in steeming what can do this PorterStemmer is if sentence is wow i loved this place then it remove i and this and take only usefull keywords and also it take love from loved like wow,love,place
    # aa koi be verb ne ana normal verb ma convert kare che means ing,s,es,ly,d,ed aa badhu remove kari dase  
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #  last 2 lines of output(review)
    # whole experi underwhelm think go ninja sushi next time
    # wast enough life pour salt wound draw time took bring check
     
    review = ' '.join(review)
    print(review)
    corpus.append(review)
print(corpus)
# print(review)
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)