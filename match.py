import joblib
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict

objects = joblib.load('data.joblib')
mid2parents = joblib.load('mid2parents.joblib')

def varlen_similarity(distances, matched_indices, n, source_confidences, hierarchy_factors):
    similarity_scores = -np.asarray(distances)*np.asarray(hierarchy_factors)
    if source_confidences is not None:
        source_confidences = np.asarray(source_confidences)[matched_indices]
    return (len(distances), np.average(similarity_scores, weights=source_confidences))

def exp_similarity(distances, matched_indices, n, source_confidences, hierarchy_factors, factor=2):
    similarity_scores = np.zeros(n)
    similarity_scores[matched_indices] = np.exp(-factor*np.asarray(distances))*np.asarray(hierarchy_factors)
    return (np.average(similarity_scores, weights=source_confidences),)

similarity_func = exp_similarity
def find_similar(boxes, labels, source_confidences=None, top_k=None, hierarchy_factor=0):
    id2scores = defaultdict(lambda:[[], [], [], [], [], []])
    for i, (box, real_label) in enumerate(zip(boxes, labels)):
        family = [real_label]
        if hierarchy_factor:
            family.extend({parent[0] for parent in mid2parents[real_label] if parent[0] in objects})
        family_df = None
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
            all_distances = (d1+d2+d3+d4)/4
            df = pd.DataFrame(np.column_stack((objects[label][1], all_distances, objects[label][0], [label]*len(objects[label][0]))))
            if family_df is None:
                family_df = df
            else:
                family_df = pd.concat([family_df, df])
        family_df = family_df.sort_values(1).drop_duplicates(0).values
        for bi, (k,distance,*b, label) in enumerate(family_df):
            id2scores[k][0].append(distance)
            id2scores[k][1].append(i) #label index
            id2scores[k][2].append(np.asarray(b)) #box
            id2scores[k][3].append(bi) #box index - for duplicate matches
            id2scores[k][4].append(1 if hindex==0 else hierarchy_factor)
            id2scores[k][5].append(label)
    result = [(k, distances, matched_indices, b, bi, label, similarity_func(distances, matched_indices, len(boxes), source_confidences, hierarchy_factors)) for k, (distances, matched_indices, b, bi, hierarchy_factors, label) in id2scores.items()]
    return sorted(result, key=lambda x: x[-1], reverse=True)[:top_k]

def filter_duplicate_boxes(values, keys):
    return list(OrderedDict(zip(keys, values)).values())

def show_results(result, labels):
    from util import draw_boxes, get_image_from_s3
    mid2label = joblib.load('mid2label.joblib')
    for k in range(len(result)):
        print(result[k][0])
        print([(mid2label[labels[i]], mid2label[l], s) for s, i, l in zip(result[k][1], result[k][2], result[k][5])])
        print([mid2label[l] for l in filter_duplicate_boxes(result[k][5], result[k][4])])
        print(result[k][6])
        image = get_image_from_s3(result[k][0])
        draw_boxes(image, result[k][3], result[k][5], show=True)

if __name__ == "__main__":
    from time import time

    boxes = [np.array([0.14549501, 0.19731247, 0.9917954 , 0.5369047 ]), np.array([0.12854484, 0.34575117, 0.6136902 , 0.7101171 ]), np.array([0.36136216, 0.1527743 , 0.9645944 , 0.37813774]), np.array([0.15144451, 0.2801198 , 0.50323135, 0.48136523]), np.array([0.37257037, 0.18959951, 0.984572  , 0.5182298 ])]
    labels = ['/m/01g317', '/m/06c54', '/m/01940j', '/m/0zvk5', '/m/09j2d']
    source_confidences = [0.9,0.8,0.7,0.6,0.5]
    top_k = 5

    start = time()
    result = find_similar(boxes, labels, source_confidences=source_confidences, top_k=top_k, hierarchy_factor=0.5)
    print('took %d sec' % (time() - start))

    show_results(result, labels)
