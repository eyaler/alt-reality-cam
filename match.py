import joblib
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import os
from time import time
from numba import jit

objects = joblib.load(os.path.join('data','data.joblib'))
mid2parents = joblib.load(os.path.join('data','mid2parents.joblib'))
mid2allsubs = joblib.load(os.path.join('data','mid2allsubs.joblib'))
mid2subcnt = joblib.load(os.path.join('data','mid2subcnt.joblib'))

@jit(nopython=True)
def varlen_similarity(distances, matched_indices, n, norm_factors):
    similarity_scores = -np.asarray(distances)/np.asarray(norm_factors)
    return similarity_scores, (len(similarity_scores), np.mean(similarity_scores))

@jit(nopython=True)
def exp_similarity(distances, matched_indices, n, norm_factors, factor=1):
    similarity_scores = np.zeros(n)
    matched_indices = np.asarray(matched_indices)
    similarity_scores[matched_indices] = np.exp(-factor*np.asarray(distances)/np.asarray(norm_factors))
    return similarity_scores[matched_indices], (np.mean(similarity_scores),)

similarity_func = exp_similarity

def box_area(box): #ymin, xmin, ymax, xmax
    return (box[...,3]-box[...,1])*(box[...,2]-box[...,0])

def find_similar(boxes, labels, source_confidences=None, hierarchy_factor=0, polypedo_discount=1, min_area=0, allowed_labels=None, allowed_ids = None, disallowed_ids = None):
    start = time()
    if allowed_ids is not None and not type(allowed_ids) in [list, tuple]:
        allowed_ids = [allowed_ids]
    if disallowed_ids is not None and not type(disallowed_ids) in [list, tuple]:
        disallowed_ids = [disallowed_ids]
    id2scores = defaultdict(list)
    for i, (box, anchor_labels) in enumerate(zip(boxes, labels)):
        family = {}
        seen = set(anchor_labels)
        for anchor_label in anchor_labels:
            family[anchor_label] = [anchor_label]
            if hierarchy_factor:
                if anchor_label in mid2allsubs:
                    sons = [label for label in mid2allsubs[anchor_label] if label not in seen and (allowed_labels is None or label in allowed_labels)]
                    family[anchor_label].extend(sons)
                    seen.update(sons)
                parents = {parent[0] for parent in mid2parents[anchor_label] if parent[1]=='Subcategory' and parent[0] not in seen and (allowed_labels is None or parent[0] in allowed_labels)}
                family[anchor_label].extend(parents)
                seen.update(parents)
        dfs = []

        for anchor_label in anchor_labels:
            for hindex, label in enumerate(family[anchor_label]):
                if label not in objects:
                    continue
                filtered_idx = box_area(objects[label][0])>min_area
                filtered_boxes = objects[label][0][filtered_idx]
                filtered_ids = objects[label][1][filtered_idx]
                #d = filtered_boxes - box # dymin, dxmin, dymax, dxmax
                # h = (filtered_boxes[:, 2] - filtered_boxes[:, 0] + box[2] - box[0]) / 2
                # w = (filtered_boxes[:, 3] - filtered_boxes[:, 1] + box[3] - box[1]) / 2
                #ymin2 = np.square(d[:, 0])  # / h
                #xmin2 = np.square(d[:, 1])  # / w
                #ymax2 = np.square(d[:, 2])  # / h
                #xmax2 = np.square(d[:, 3])  # / w

                all_distances = np.sqrt(np.mean(np.square(filtered_boxes - box), axis=1))
                norm_factor = (1 if source_confidences is None else source_confidences[i]) * (hierarchy_factor if hindex>0 else 1) * polypedo_discount ** max(mid2subcnt[label], mid2subcnt[anchor_label]) #note here the first anchor label will be used
                dfs.append(pd.DataFrame({'image_id':filtered_ids, 'distance':all_distances, 'label_index':i, 'ymin':filtered_boxes[:,0], 'xmin':filtered_boxes[:,1], 'ymax':filtered_boxes[:,2], 'xmax':filtered_boxes[:,3], 'box_id':range(len(filtered_boxes)), 'label':label, 'anchor_label':anchor_label, 'norm_factor':norm_factor}, columns=['image_id','distance','label_index','ymin','xmin','ymax','xmax','box_id', 'label', 'anchor_label', 'norm_factor']))

        family_df = pd.concat(df for df in dfs)
        family_df = family_df.sort_values('distance').drop_duplicates('image_id')

        for image_id, *row in family_df.values:
           id2scores[image_id].append(row)

    id2scores = zip(id2scores.keys(), [zip(*rows) for rows in id2scores.values()])
    result = [(image_id, matched_indices, np.asarray(list(zip(ymin,xmin,ymax,xmax))), box_id, labels, filter_duplicate_boxes(box_id, labels), anchor_labels, filter_duplicate_boxes(box_id, anchor_labels), *similarity_func(distances, matched_indices, len(boxes), norm_factors)) for image_id, (distances, matched_indices, ymin,xmin,ymax,xmax, box_id, labels, anchor_labels, norm_factors) in id2scores if (allowed_ids is None or image_id in allowed_ids) and (disallowed_ids is None or image_id not in disallowed_ids)]

    print('matching took %d sec' % (time() - start))
    return sorted(result, key=lambda x: x[-1], reverse=True)

def filter_duplicate_boxes(keys, values):
    return list(OrderedDict(zip(keys, values)).values())

def show_results(result, labels, scores=None, show_size=False):
    from util import draw_boxes, get_image_from_s3
    mid2label = joblib.load(os.path.join('data','mid2label.joblib'))
    for k in range(len(result)):
        print(result[k][0])
        print(sorted([(mid2label[labels[matched_index]], mid2label[label], score) for score, matched_index, label in zip(result[k][8], result[k][1], result[k][4])], key=lambda x: -x[2]))
        print([mid2label[label] for label in result[k][5]])
        print(result[k][9])
        image = get_image_from_s3(result[k][0])
        draw_boxes(image, result[k][2], result[k][4], scores=scores, uid=result[k][3], show=True, show_size=show_size)

if __name__ == "__main__":
    boxes = [np.array([0.14549501, 0.19731247, 0.9917954 , 0.5369047 ]), np.array([0.12854484, 0.34575117, 0.6136902 , 0.7101171 ]), np.array([0.36136216, 0.1527743 , 0.9645944 , 0.37813774]), np.array([0.15144451, 0.2801198 , 0.50323135, 0.48136523]), np.array([0.37257037, 0.18959951, 0.984572  , 0.5182298 ])]
    labels = ['/m/01g317', '/m/06c54', '/m/01940j', '/m/0zvk5', '/m/09j2d']
    source_confidences = None
    top_k = 5
    polypedo_discount = 1
    hierarchy_factor = 0.5
    result = find_similar(boxes, labels, source_confidences=source_confidences, hierarchy_factor=hierarchy_factor, polypedo_discount=polypedo_discount)[:top_k]
    show_results(result, labels, scores=source_confidences, show_size=True)
