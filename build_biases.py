# using https://github.com/RaRe-Technologies/gensim-data

import pandas as pd
import numpy as np
import joblib
import os

label2mid = joblib.load(os.path.join('data','label2mid.joblib'))

def get_manual_bias_labels(biases):
    df = pd.read_csv('bias_labels.csv')
    biases = [bias if type(bias)==str else bias[0] for bias in biases]
    dd = {bias: [label for label in df[bias] if label is not np.nan] for bias in biases if bias!='zero'}
    assert {label for sublist in dd.values() for label in sublist} <= set(label2mid)
    return dd


if __name__ == '__main__':
    import gensim.downloader as api
    from gensim.models import KeyedVectors

    threshold = 0.4
    topk = 3
    sim = 'manual' #'pairdirection', '3cosadd', '3cosmul', '2cosadd', '2cosmul', 'manual', 'manualreject'
    find_missing = True

    biases = [('zero',), ('gender','woman','man'), ('war','war','peace'), ('money','rich','poor'), ('love','love','hate'), ('fear',)]

    model_name, prefix = 'fasttext-wiki-news-subwords-300', ''
    #model_name, prefix = 'glove-wiki-gigaword-300', ''
    #model_name, prefix = 'conceptnet-numberbatch-17-06-300', '/c/en/'
    #model_name, prefix = 'word2vec-google-news-300', ''

    model_vectors = api.load(model_name)  # load pre-trained word-vectors from gensim-data
    print('model loaded')

    def norm(label):
        return prefix + label.lower().split(' (')[0].replace(' ', '-')
    norm_labels = [norm(label) for label in label2mid]

    secondary_labels = []
    def compound_norm(label):
        global missing_map
        norm_label = norm(label)
        if find_missing and norm_label not in model_vectors and '-' in norm_label:
            candidate = norm_label.split('-')[1]
            if candidate in model_vectors:
                norm_label = candidate
                if norm_label in norm_labels and label not in secondary_labels:
                    secondary_labels.append(label)
        return norm_label

    labels, vectors = [list(a) for a in zip(*[(label, model_vectors[compound_norm(label)]) for label in label2mid if compound_norm(label) in model_vectors])]
    assert len(labels)==len(set(labels))
    print ('found %d out of %d labels in vocab'%(len(labels), len(label2mid)))
    if len(labels)<len(label2mid):
        print('missing labels:', sorted(label for label in label2mid if label not in labels))
    uniques = len(set(tuple(vector) for vector in vectors))
    if len(labels)>uniques:
        print('note: there are %d duplicate vectors'%(len(vectors)-uniques))
        print('secondary labels:',secondary_labels)

    shifts, shift_vectors = [list(a) for a in zip(*[(label, model_vectors[compound_norm(label)]) for sublist in biases for label in sublist[1:] if label not in labels])]

    word_vectors = KeyedVectors(model_vectors.vector_size)
    word_vectors.add(labels+shifts, vectors+shift_vectors)

    def get_auto_bias_labels(biases, topn=None):
        return {bias[0]: word_vectors.similar_by_vector(word_vectors[bias[1]]-word_vectors[bias[2]], topn=len(labels), restrict_vocab=len(labels))[:topn] for bias in biases if bias!='zero'}
        #note: effectively this could be less than topn due to possible secondary labels

    def pair_direction(positive, negative, topn=None):
        vocab = [word for word in word_vectors.vocab if word not in positive+negative+shifts+secondary_labels]
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
            biased = [bias[0]]
        else:
            biased = func(positive=[label, bias[0]], negative=None if sim.startswith('2') else [bias[1]], topn=(topk+len(secondary_labels) if threshold is None else len(labels))+len(shifts))
            biased = [biased_label[0] for biased_label in biased if biased_label[0] not in shifts+secondary_labels and (threshold is None or word_vectors.similarity(label, biased_label[0]) >= threshold)]
            biased = biased[:topk] if len(biased)>0 else label
        return biased

    rows = []
    bias_labels = get_manual_bias_labels(biases)
    with open('bias_scores.txt', 'w') as f:
        for label in labels:
            row = []
            for bias in biases:
                biased = [label]
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
                    scores, vocab = zip(*sorted(zip(scores, vocab), key=lambda x: (-x[0], label!=x[1], x[1] in secondary_labels))[:topk])
                    f.write('%s %s '%(label,bias[0])+' '.join('%.4f %s'%(scores[k],vocab[k]) for k in range(len(vocab))) + '\n')
                    if scores[0] >= threshold:
                        biased = [v for v,s in zip(vocab,scores) if s>threshold]
                elif len(bias)>1:
                    biased = get_bias(label, bias[1:], sim, threshold)
                row.append('/'.join(biased))
            rows.append(row)

    pd.DataFrame(rows, columns=[bias[0] for bias in biases]).to_csv('biases.csv', index=False)
