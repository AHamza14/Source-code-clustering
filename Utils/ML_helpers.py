import os
import re
import nltk
import pandas as pd

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from random import randrange

# word embeddings
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
# import seaborn as sns
nltk.download('stopwords')
nltk.download('punkt_tab')

# Initialize the stemmer and stop words
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


def preprocess(text):
    text_input = re.sub('[^a-zA-Z1-9]+', ' ', str(text))
    output = re.sub(r'\d+', '', text_input)
    return output.lower().strip()


def remove_stopwords(text):
    filtered_words = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(filtered_words)


def build_corpus(data):
    corpus = []
    for sentence in data:
        word_list = sentence.split(" ")
        corpus.append(word_list)
    return corpus

# Old code

def Load_data_set(path):
    source_code_data = []
    files_name = []
    for subdir, dirs, files in os.walk(path):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith(".cs"):
                f = open(filepath)
                # f = open(filename, 'rb')
                s = f.read()
                source_code_data.append(s)
                files_name.append(os.path.basename(f.name.replace(".cs", "")))
                f.close()
    return files_name, source_code_data


def data_preprocessing(data, type_preprocessing):
    # remove programming language keywords
    data = pd.Series(data).replace(
        r'\b(using|\.|break|true|,|"|\'|\(|\)|\{|\}|;|\[]|\[|\]|\+|\-|\*|=|false|object|private|sender|interface|enum|namespace|abstract|switch|case|if|else|for|null|void|double|int|using|string|return|new|var|public|static|class)|(\w+)\1+\b',
        ' ', regex=True)

    # Split CamelCase
    data = split_camel_case(data)

    # remove stop words
    data = remove_stopwords(data)

    # stem words
    if type_preprocessing == "PPwS":
        data = stem_data(data)

    # new_file = ""  # for ff in x:  #     new_file = new_file + ff  # new_data.append(new_file)
    # data = [finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', x) for x in data]
    return data


def remove_stopwords(sentences):
    processed_sentences = []

    for sentence in sentences:
        # Tokenize the sentence
        words = word_tokenize(sentence)

        # Remove stop words
        filtered_words = [word for word in words if word.lower() not in stop_words]

        # Join the filtered words back into a sentence and add to result list
        processed_sentences.append(' '.join(filtered_words))

    return processed_sentences


def stem_data(sentences):
    processed_sentences = []

    for sentence in sentences:
        # Tokenize the sentence
        words = word_tokenize(sentence)

        # Stem the words
        stemmed_words = [ps.stem(word) for word in words]

        # Join the filtered words back into a sentence and add to result list
        processed_sentences.append(' '.join(stemmed_words))

    return processed_sentences


def data_noise(data):
    new_data = []
    for file in data:
        new_file = ''
        for word in file.split():
            # swab chars
            rand1 = randrange(len(word)) - 1
            rand2 = 0 if rand1 == len(word) else rand1 + 1
            lst = list(word)
            lst[rand1], lst[rand2] = lst[rand2], lst[rand1]
            modifiedWord = ''.join(lst)

            # remove char
            i = randrange(len(modifiedWord))
            modifiedWord = modifiedWord[:i] + modifiedWord[i + 1:]

            new_file += " " + modifiedWord
        new_data.append(new_file)
    return new_data


def split_camel_case(data):
    new_data = []
    for file in data:
        splitted = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', file)).split()
        splitted = " ".join(splitted)
        new_data.append(
            splitted.replace('{', ' ').replace('}', ' ').replace('(', ' ').replace(')', ' ').replace(';', ' ').replace(
                ':', ' '))

    return new_data


def data_vectorization(cleaned_data):
    # vectorizer = TfidfVectorizer(stop_words={'english'})
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(cleaned_data)


def elbow_method(X, clusters_number):
    Sum_of_squared_distances = []
    K = range(2, clusters_number)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(X)
        Sum_of_squared_distances.append(km.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def silhouette_method(X, clusters_number):
    M = range(2, clusters_number)
    silhouette_scores = []
    for k in M:
        kms = KMeans(n_clusters=k)
        kms = kms.fit(X)
        score = silhouette_score(X, kms.labels_, metric='euclidean')
        silhouette_scores.append(score)

    print("Highest silhouette score is: " + str(silhouette_scores.index(max(silhouette_scores))) + " -> " + str(
        max(silhouette_scores)))
    plt.plot(M, silhouette_scores, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.title('Silhouette score method')
    plt.show()
    return silhouette_scores.index(max(silhouette_scores))


def knn(X):
    neighbors = 2
    # X_embedded is your data
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(X)
    distances, indices = nbrs.kneighbors(X)
    distance_desc = sorted(distances[:, indices - 1], reverse=True)

    x = list(range(1, len(distance_desc) + 1))
    plt.plot(x, distance_desc, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.title('Knn method')
    plt.show()
