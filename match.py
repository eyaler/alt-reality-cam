import joblib
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict

objects = joblib.load('data.joblib')
mid2parents = joblib.load('mid2parents.joblib')

def varlen_similarity(s, i, n, scores, hfactors):
    weights = hfactors
    if scores is not None:
        weights *= np.asarray(scores)[i]
    return (len(s), -np.average(s, weights=weights))

def exp_similarity(s, i, n, scores, hfactors, factor=2):
    sims = np.zeros(n)
    sims[i] = np.exp(-factor*np.asarray(s))
    weights = np.zeros(n)
    weights[i] = hfactors
    if scores is not None:
        weights *= scores
    return (np.average(sims, weights=weights),)

similarity_func = exp_similarity
def find_similar(boxes, labels, scores=None, top_k=None, hierarchy_factor=0):
    id2scores = defaultdict(lambda:[[], [], [], [], []])
    for i, (box, real_label) in enumerate(zip(boxes, labels)):
        family = [real_label]
        if hierarchy_factor:
            family.extend({parent[0] for parent in mid2parents[real_label] if parent[0] in objects})
        for hindex,label in enumerate(family):
            d = objects[label][0] - box #dymin, dxmin, dymax, dxmax
            h = (objects[label][0][:, 2] - objects[label][0][:, 0] + box[2] - box[0]) / 2
            w = (objects[label][0][:, 3] - objects[label][0][:, 1] + box[3] - box[1]) / 2
            nymin = d[:, 0] / h
            nxmin = d[:, 1] / w
            nymax = d[:, 2] / h
            nxmax = d[:, 3] / w
            d1 = np.sqrt(np.square(nymin)+np.square(nxmin))
            d2 = np.sqrt(np.square(nymin)+np.square(nxmax))
            d3 = np.sqrt(np.square(nymax)+np.square(nxmax))
            d4 = np.sqrt(np.square(nymax)+np.square(nxmin))
            dist = (d1+d2+d3+d4)/4
            df = pd.DataFrame(np.column_stack((objects[label][1], dist, objects[label][0]))).sort_values(1).drop_duplicates(0).values
            for bi, (k,s,*b) in enumerate(df):
                id2scores[k][0].append(s) #scores
                id2scores[k][1].append(i) #label index
                id2scores[k][2].append(np.asarray(b)) #box
                id2scores[k][3].append(bi) #box index - for duplicate matches
                id2scores[k][4].append(1 if hindex==0 else hierarchy_factor) # hierarchy factor
    result = [(k, s, i, b, bi, similarity_func(s, i, len(boxes), scores, hfactors)) for k, (s, i, b, bi, hfactors) in id2scores.items()]
    return sorted(result, key=lambda x: x[5], reverse=True)[:top_k]

def filter_duplicate_boxes(label_indices, box_indices):
    return list(OrderedDict(zip(zip(box_indices, label_indices), label_indices)).values())

def show_results(result, labels):
    from util import draw_boxes, get_image_from_s3
    mid2label = joblib.load('mid2label.joblib')
    for k in range(len(result)):
        print(result[k][0])
        print([(mid2label[labels[i]], s) for s,i in zip(*result[k][1:3])])
        print([mid2label[labels[i]] for i in filter_duplicate_boxes(result[k][2], result[k][4])])
        print(result[k][5])
        image = get_image_from_s3(result[k][0])
        draw_boxes(image, result[k][3], [labels[i] for i in result[k][2]], show=True)


if __name__ == "__main__":
    from time import time

    boxes = [np.array([0.14549501, 0.19731247, 0.9917954 , 0.5369047 ]), np.array([0.12854484, 0.34575117, 0.6136902 , 0.7101171 ]), np.array([0.36136216, 0.1527743 , 0.9645944 , 0.37813774]), np.array([0.15144451, 0.2801198 , 0.50323135, 0.48136523]), np.array([0.37257037, 0.18959951, 0.984572  , 0.5182298 ])]
    labels = ['/m/01g317', '/m/06c54', '/m/01940j', '/m/0zvk5', '/m/09j2d']
    scores = [0.9,0.8,0.7,0.6,0.5]
    top_k = 5

    start = time()
    result = find_similar(boxes, labels, scores=scores, top_k=top_k, hierarchy_factor=0)
    print('took %d sec' % (time() - start))

    show_results(result, labels)