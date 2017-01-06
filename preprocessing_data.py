import os, string, pickle, re, random
import numpy as np
import pandas as pd

from collections import Counter, deque
from nltk import word_tokenize

def remove_punctuation(sentence):
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub('', sentence)

def custom_tokenizer(doc):
    doc = remove_punctuation(doc)
    doc = doc.lower()
    return doc.split()

def time_format(second):
    h = second // 3600
    m = ((second % 3600) // 60)
    s = second % 60
    return [h, m, s]

def Eu_dist(a, b):
    return np.linalg.norm(a - b)


def Cos_dist(a, b):
    return 1 - a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))

def load_data(pickle_in):
    with open(pickle_in, 'rb') as f:
        contents = pickle.load(f)
    return contents

def create_docs_pickle(fin, fout):
    print('Run create_docs_pickle(', fin, ',', fout, ')')
    docs = []
    with open(fin, 'r') as f:
        for line in f:
            doc = line.split(":::b'")[1]
            docs.append([w for w in word_tokenize(doc)])

    with open(fout, 'wb') as f:
        pickle.dump(docs, f)

    return docs

def build_dataset(words, vocab_size=30000):
    count = [['UNK', -1]]
    count.extend(Counter(words).most_common(vocab_size-1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0

    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)

    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reverse_dictionary

def build_doc_dataset(docs, vocab_size=30000, doc_dataset_pickle='docs_dataset.pickle'):
    print('Run build_doc_dataset(...', doc_dataset_pickle, ')')
    words = []
    doc_idxs = []
    for i, doc in enumerate(docs):
        words.extend(doc)
        doc_idxs.extend([i] * len(doc))

    word_idx, count, dictionary, reverse_dictionary = build_dataset(words, vocab_size=vocab_size)
    with open(doc_dataset_pickle, 'wb') as f:
        pickle.dump((doc_idxs, word_idx, count, dictionary, reverse_dictionary), f)

    return doc_idxs, word_idx, count, dictionary, reverse_dictionary

def process_all_words(train_pos, train_neg, test_pos, test_neg, fout):
    print('Run process_all_words(', train_pos, ',', train_neg, ',', test_pos, ',', test_neg, ')')
    outfile = open(fout, 'a')

    with open(train_pos, 'rb') as f:
        for i, words in enumerate(f):
            outline = 'train_pos_' + str(i) + ':::' + str(words)
            outfile.write('%s\n' % outline)
    with open(train_neg, 'rb') as f:
        for i, words in enumerate(f):
            outline = 'train_neg_' + str(i) + ':::' + str(words)
            outfile.write('%s\n' % outline)
    with open(test_pos, 'rb') as f:
        for i, words in enumerate(f):
            outline = 'test_pos_' + str(i) + ':::' + str(words)
            outfile.write('%s\n' % outline)
    with open(test_neg, 'rb') as f:
        for i, words in enumerate(f):
            outline = 'test_neg_' + str(i) + ':::' + str(words)
            outfile.write('%s\n' % outline)

    outfile.close()

def shuffle(fin, fout):
    print('Run shuffle(', fin, ',', fout, ')')
    df = pd.read_csv(fin, encoding='latin-1', error_bad_lines=False)
    df = df.iloc[np.random.permutation(len(df))]
    print(df.head())
    df.to_csv(fout, index=False)

def create_train_test_data(docs_embeddings_pickle, data_file, test_size=0.1):
    print('Run create_docs_pickle(', docs_embeddings_pickle, ',', data_file, ')')
    train_inputs = []
    train_labels = []
    test_inputs = []
    test_labels = []

    with open(docs_embeddings_pickle, 'rb') as f:
        docs_embeddings = pickle.load(f)

    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('train_pos') or line.startswith('test_pos'):
                if np.random.rand() < test_size:
                    test_labels.append('1')
                    test_inputs.append(docs_embeddings[i])
                else:
                    train_labels.append('1')
                    train_inputs.append(docs_embeddings[i])
            if line.startswith('train_neg') or line.startswith('test_neg'):
                if np.random.rand() < test_size:
                    test_labels.append('0')
                    test_inputs.append(docs_embeddings[i])
                else:
                    train_labels.append('0')
                    train_inputs.append(docs_embeddings[i])


    return train_inputs, train_labels, test_inputs, test_labels

def process_new_docs(docs, dictionary, next_doc_idx):
    print('Run process_new_doc(...)')

    new_word_idx = list()
    new_doc_idx = list()
    len_doc = 0

    for i, doc in enumerate(docs):
        doc = custom_tokenizer(doc)
        new_doc_idx.extend([i + next_doc_idx] * len(doc))
        for word in doc:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
            new_word_idx.append(index)

        len_doc = i+1

    return new_word_idx, new_doc_idx, len_doc