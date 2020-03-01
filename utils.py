import nltk
import pickle
import re
import numpy as np


nltk.download('stopwords')
from nltk.corpus import stopwords


def text_prepare(text):
    """Performs simple text preprocessing."""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file."""

    ss_embeddings = {}
    
    with open(embeddings_path, 'r') as tsv:
        for line in tsv:
            line = line.split('\t')
            ss_embeddings[line[0]] = np.array(line[1:], dtype=np.float32)
        embeddings_dim = len(line[1:])
        
    return ss_embeddings, embeddings_dim


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""

    question = question.split(' ')
    question_mean = np.zeros(dim)
    count = 0

    for word in question:
        try:
            word_embed = embeddings[word]
            question_mean += word_embed
            count += 1
        except KeyError:
            pass
    
    if count == 0:
        return question_mean
    
    return (question_mean / count).astype(np.float16)


def unpickle_file(filename):
    """Returns the model artifacts by unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
