import pandas as pd
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

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

    # n-gram
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i+gram])]
        words = w
        
    return words

def process_content(data):
    data.apply(lambda row : process_comment(row))

df = read_data(DATA_PATH)
X = df['CONTENT']
y = df['CLASS']

process_content(X)