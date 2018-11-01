import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

DATA_PATH = '.\\data'

def read_youtube_dataset(file_path):
    # read dataset and get rid of the COMMENT_ID, AUTHOR and DATE
    data = pd.read_csv(file_path)
    data.drop(labels=['COMMENT_ID', 'AUTHOR', 'DATE'], axis=1, inplace=True)
    data['CLASS'] = data['CLASS'].astype('category')
    return data

def read_data(path):
    files = os.listdir(path)

    datasets = []
    for f in files:
        file_path = os.path.join(path, f)
        if os.path.isfile(file_path):
            datasets.append(read_youtube_dataset(file_path))
    
    return pd.concat(datasets)

def process_comment(comment, gram=2):
    # convert to lowercase (if required)
    comment = comment.lower()
    
    # tokenize the message
    words = word_tokenize(comment)

    # remove stopwords
    sw = stopwords.words('english')
    words = [word for word in words if word not in sw]

    # stemmer
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

def process_content(data):
    return data.apply(lambda row : process_comment(row))

df = read_data(DATA_PATH)
content = process_content(df['CONTENT'])
y = df['CLASS']

vectorizer = TfidfVectorizer('english')
X = vectorizer.fit_transform(content)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

mnb = MultinomialNB()
mnb_parameters = {
    'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'fit_prior': [True, False]
}

clf = GridSearchCV(mnb, mnb_parameters, cv=5, n_jobs=-1)
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.score(X_test, y_test))

svc = SVC(gamma='scale')
svc_parameters = {
    'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
    'C': [1.0, 2.0, 3.0, 5.0, 10.0]
}

clf = GridSearchCV(svc, svc_parameters, cv=5, n_jobs=-1)
clf.fit(X_train, y_train)
print(clf.best_params_)
print(clf.score(X_test, y_test))
