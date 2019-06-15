# using https://github.com/RaRe-Technologies/gensim-data

import gensim.downloader as api
import joblib
import os

label2mid = joblib.load(os.path.join('data','label2mid.joblib'))

biases = {'zero':[], 'gender':['woman','man'], 'military':['war','peace'], 'money':['rich','pool'], 'love':['love','hate']}

word_vectors = api.load("conceptnet-numberbatch-17-06-300")  # load pre-trained word-vectors from gensim-data

exit()
for label in label2mid:
    pass

#case

print(word_vectors.most_similar(positive=['woman', 'king'], negative=['man']))
print(word_vectors.most_similar_cosmul(positive=['woman', 'king'], negative=['man']))

#bias_map = pd.read_csv('biases.csv', index_col='zero').apply(lambda x: x.str.capitalize())
