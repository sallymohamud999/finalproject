#this imports the pandas library, allowing us to access files and edit file content
import pandas as pd

#stores and reads the true and fake files into their respective variables
true = pd.read_csv('True20.csv')
fake = pd.read_csv('Fake20.csv')

#reads the top 3 rows in the files
true.head(3)
fake.head(3)

#gives us detail on the shape on the data [rows, columns]
true.shape
fake.shape

#assigning labels for websites that will be real or fake
true['label'] = 1
fake['label'] = 0

#use the first 5000 data of true and fake dataset for building the model
frames = [true.loc[:20][:], fake.loc[:20][:]]

#concats the frames
df = pd.concat(frames)

df.shape

df.tail()

X = df.drop('label', axis=1)
y = df['label']

df = df.dropna()
df2 = df.copy()

df2.reset_index(inplace=True)
df2.head()
df2['title'][2]

##DATA PREPROCESSONING
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

import reimport nltk
nltk.downloads('stopwords')

corpus = []
for i in range (0, len(df2)):
    review = re.sub('[^a-zA-Z]', ' ', df['text'][i])
    review = review.lower
    review = review.split

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#TFidf Vectoriser
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

##MODEL BUILDING - PASSIVE AGGRESSIVE CLASSIFIER
from sklearn.linear_model import PassiveAggressive Classifier
classifier = PassiveAggressiveClassifier(max_iter=1000)

from sklearn import metrics
import numpy as np
import itertools

classifier.fit(X_train, y_train)

pred = classifier.predict(X_test)

score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

##VALIDATE ON AN UNSEEN DATAPOINT
review = re.sub('[^a-zA-Z]', ' ', fake['test'][13070])
review = review.lower()
review = review.split()

review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
review = ' '.join(review)
review

val = tfidf_v.transform([review]).toarray()
classifier.predict(val)

##SAVE MODEL AND VECTORISE
import pickle
pickle.dump(classifier, open('model2.pkl', 'wb'))
pickle.dump(tfidf_v, open('tfidfvect2.pkl', 'wb'))

##LOAD MODEL AND VECTORISE TO PREDIC THE PREVIOUS DATA POINT
joblib_model = pickle.load(open('model2.pkl', 'rb'))
joblib_vect = pickle.load(open('tfidfvect2.pkl', 'rb'))
val_pkl = joblib_vect.transform([review]).toarray()
joblic_model.predict(val_pkl)