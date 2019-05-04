import joblib
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import os
from time import time

objects = joblib.load(os.path.join('data','data.joblib'))
mid2parents = joblib.load(os.path.join('data','mid2parents.joblib'))
mid2children = joblib.load(os.path.join('data','mid2children.joblib'))

def varlen_similarity(distances, matched_indices, n, source_confidences, hierarchy_factors):
    similarity_scores = -np.asarray(distances)/np.asarray(hierarchy_factors)
    if source_confidences is not None:
        similarity_scores /= np.asarray(source_confidences)[matched_indices]
    return similarity_scores, (len(similarity_scores), np.mean(similarity_scores))

def exp_similarity(distances, matched_indices, n, source_confidences, hierarchy_factors, factor=1):
    similarity_scores = np.zeros(n)
    conf = 1
    if source_confidences is not None:
        conf = np.asarray(source_confidences)[matched_indices]
    similarity_scores[matched_indices] = np.exp(-factor*np.asarray(distances)/np.asarray(hierarchy_factors)/conf)
    return similarity_scores[matched_indices], (np.mean(similarity_scores),)

similarity_func = exp_similarity
def find_similar(boxes, labels, source_confidences=None, top_k=None, hierarchy_factor=None, allowed_ids = None):
    start = time()
    if allowed_ids is not None and not type(allowed_ids) in [list, tuple]:
            allowed_ids = [allowed_ids]
    id2scores = defaultdict(lambda:[[], [], [], [], [], []])
    for i, (box, real_label) in enumerate(zip(boxes, labels)):
        family = [real_label]
        if hierarchy_factor:
            family.extend({parent[0] for parent in mid2parents[real_label] if parent[0] in objects and parent[1]=='Subcategory'})
            if real_label in mid2children:
                family.extend({child[0] for child in mid2children[real_label] if child[1]=='Subcategory'})
        family_df = None
        for hindex, label in enumerate(family):
            d = objects[label][0] - box #dymin, dxmin, dymax, dxmax
            #h = (objects[label][0][:, 2] - objects[label][0][:, 0] + box[2] - box[0]) / 2
            #w = (objects[label][0][:, 3] - objects[label][0][:, 1] + box[3] - box[1]) / 2
            nymin = d[:, 0] #/ h
            nxmin = d[:, 1] #/ w
            nymax = d[:, 2] #/ h
            nxmax = d[:, 3] #/ w
            d1 = np.sqrt(np.square(nymin)+np.square(nxmin))
            d2 = np.sqrt(np.square(nymin)+np.square(nxmax))
            d3 = np.sqrt(np.square(nymax)+np.square(nxmax))
            d4 = np.sqrt(np.square(nymax)+np.square(nxmin))
            all_distances = (d1+d2+d3+d4)/4
            length = len(objects[label][0])
            df = pd.DataFrame(np.column_stack((objects[label][1], all_distances, objects[label][0], range(length), [label]*length, [hindex]*length)))
            if family_df is None:
                family_df = df
            else:
                family_df = pd.concat([family_df, df])
        family_df = family_df.sort_values(1).drop_duplicates(0).values
        for k,distance,*b, uid, label, hindex in family_df:
            id2scores[k][0].append(distance)
            id2scores[k][1].append(i) #label index
            id2scores[k][2].append(np.asarray(b)) #box
            id2scores[k][3].append(uid) #box index - for duplicate matches
            id2scores[k][4].append(1 if hindex==0 else hierarchy_factor)
            id2scores[k][5].append(label)
    result = [(k, matched_indices, box, uid, label, *similarity_func(distances, matched_indices, len(boxes), source_confidences, hierarchy_factors)) for k, (distances, matched_indices, box, uid, hierarchy_factors, label) in id2scores.items() if allowed_ids is None or k in allowed_ids]
    print('took %d sec' % (time() - start))
    return sorted(result, key=lambda x: x[-1], reverse=True)[:top_k]

def filter_duplicate_boxes(keys, values):
    return list(OrderedDict(zip(keys, values)).values())

def show_results(result, labels):
    from util import draw_boxes, get_image_from_s3
    mid2label = joblib.load(os.path.join('data','mid2label.joblib'))
    for k in range(len(result)):
        print(result[k][0])
        print(sorted([(mid2label[labels[matched_indices]], mid2label[label], scores) for scores, matched_indices, label in zip(result[k][5], result[k][1], result[k][4])], key=lambda x: -x[2]))
        print([mid2label[label] for label in filter_duplicate_boxes(result[k][3], result[k][4])])
        print(result[k][6])
        image = get_image_from_s3(result[k][0])
        draw_boxes(image, result[k][2], result[k][4], uid=result[k][3], show=True)

if __name__ == "__main__":
    boxes = [np.array([0.14549501, 0.19731247, 0.9917954 , 0.5369047 ]), np.array([0.12854484, 0.34575117, 0.6136902 , 0.7101171 ]), np.array([0.36136216, 0.1527743 , 0.9645944 , 0.37813774]), np.array([0.15144451, 0.2801198 , 0.50323135, 0.48136523]), np.array([0.37257037, 0.18959951, 0.984572  , 0.5182298 ])]
    labels = ['/m/01g317', '/m/06c54', '/m/01940j', '/m/0zvk5', '/m/09j2d']
    source_confidences = [1,1,1,1,1]
    top_k = 5

    result = find_similar(boxes, labels, source_confidences=source_confidences, top_k=top_k, hierarchy_factor=0.5)

    show_results(result, labels)
