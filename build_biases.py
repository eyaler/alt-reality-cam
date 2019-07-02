# using https://github.com/RaRe-Technologies/gensim-data

import gensim.downloader as api
from gensim.models import KeyedVectors
import joblib
import os
import pandas as pd
import numpy as np

threshold = 0.4
sim = 'manual' #'pairdirection', '3cosadd', '3cosmul', '2cosadd', '2cosmul', 'manual', 'manualreject'

biases = [('zero',), ('gender','woman','man'), ('military','war','peace'), ('money','rich','poor'), ('love','love','hate'), ('fear',)]

model_name, prefix = 'fasttext-wiki-news-subwords-300', ''
#model_name, prefix = 'glove-wiki-gigaword-300', ''
#model_name, prefix = 'conceptnet-numberbatch-17-06-300', '/c/en/'
#model_name, prefix = 'word2vec-google-news-300', ''

label2mid = joblib.load(os.path.join('data','label2mid.joblib'))

model_vectors = api.load(model_name)  # load pre-trained word-vectors from gensim-data
print('model loaded')

def norm(label):
    return prefix + label.lower().split(' (')[0].replace(' ', '-')

labels, vectors = zip(*[(label, model_vectors[norm(label)]) for label in label2mid if norm(label) in model_vectors])
assert len(labels)==len(set(labels))
print ('found %d out of %d labels in vocab'%(len(labels), len(label2mid)))
if len(labels)<len(label2mid):
    print('missing labels:', sorted(label for label in label2mid if label not in labels))

shifts, shift_vectors = zip(*[(label, model_vectors[norm(label)]) for sublist in biases for label in sublist[1:] if label not in labels])

word_vectors = KeyedVectors(model_vectors.vector_size)
word_vectors.add(labels+shifts, list(vectors+shift_vectors))

def get_auto_bias_labels(topn=None):
    return {bias[0]: word_vectors.similar_by_vector(word_vectors[bias[1]]-word_vectors[bias[2]], topn=len(labels), restrict_vocab=len(labels)[:topn]) for bias in biases[1:]}

def get_manual_bias_labels():
    df = pd.read_csv('bias_labels.csv')
    dd = {bias[0]: [label for label in df[bias[0]] if label is not np.nan] for bias in biases[1:]}
    assert {label for sublist in dd.values() for label in sublist} <= set(label2mid)
    return dd

def pair_direction(positive, negative, topn=None):
    vocab = [word for word in word_vectors.vocab if word not in positive+negative+list(shifts)]
    vector_1 = word_vectors[positive[1]] - word_vectors[negative[0]]
    vectors_all = np.asarray([word_vectors[word] - word_vectors[positive[0]] for word in vocab])
    scores = word_vectors.cosine_similarities(vector_1, vectors_all)
    ind = np.argsort(-scores)
    return np.asarray(list(zip(vocab,scores)))[ind][:topn].tolist()

def get_bias(label, bias, sim, threshold=None):
    func = word_vectors.most_similar
    if sim == 'pairdirection':
        func = pair_direction
    elif sim.endswith('mul'):
        func = word_vectors.most_similar_cosmul
    if not sim.startswith('2') and len(bias)>1 and bias[0] in labels and np.allclose(word_vectors[label], word_vectors[bias[1]]):
        biased = bias[0]
    else:
        biased = func(positive=[label, bias[0]], negative=None if sim.startswith('2') else [bias[1]], topn=(1 if threshold is None else len(labels))+len(shifts))
        biased = [biased_label[0] for biased_label in biased if biased_label[0] not in shifts and (threshold is None or word_vectors.similarity(label, biased_label[0]) >= threshold)]
        biased = biased[0] if len(biased)>0 else label
    return biased

rows = []
bias_labels = get_manual_bias_labels()
with open('bias_scores.txt', 'w') as f:
    for label in labels:
        row = []
        for bias in biases:
            biased = label
            if sim.startswith('manual') and bias[0]!='zero':
                vocab = [word for word in bias_labels[bias[0]] if word in labels]
                vectors_all = np.asarray([word_vectors[word] for word in vocab])
                vector_1 = word_vectors[label]
                if sim.endswith('reject') and len(bias)>2:
                    bias_vector = word_vectors[bias[1]]-word_vectors[bias[2]]
                    bias_vector /= np.linalg.norm(bias_vector)
                    vectors_all -= np.outer(np.dot(vectors_all, bias_vector), bias_vector)
                    vector_1 = vector_1 - np.dot(vector_1, bias_vector)*bias_vector
                scores = word_vectors.cosine_similarities(vector_1, vectors_all)
                f.write('%s %s %.4f %s\n'%(label,bias[0],np.max(scores),vocab[np.argmax(scores)]))
                if np.max(scores) >= threshold:
                    biased = vocab[np.argmax(scores)]
            elif len(bias)>1:
                biased = get_bias(label, bias[1:], sim, threshold)
            row.append(biased)
        rows.append(row)

pd.DataFrame(rows, columns=[bias[0] for bias in biases]).to_csv('biases.csv', index=False)
