import pandas as pd
import numpy as np
import joblib
import json
from collections import defaultdict
import os

os.makedirs('data', exist_ok=True)

def rotate(b, r):
    # counter clockwise!
    if r==270:
        return [1-b[3],1-b[2],b[0],b[1]]
    elif r==180:
        return [1-b[1],1-b[0],1-b[3],1-b[2]]
    elif r==90:
        return [b[2],b[3],1-b[1],1-b[0]]
    return b #xmin, xmax, ymin, ymax

def fix_order(b):
    return [b[2], b[0], b[3], b[1]] #ymin, xmin, ymax, xmax

df1 = pd.read_csv(os.path.join('open_images', 'train-annotations-bbox.csv'))
df2 = pd.read_csv(os.path.join('open_images', 'validation-annotations-bbox.csv'))
df3 = pd.read_csv(os.path.join('open_images', 'test-annotations-bbox.csv'))
data = pd.concat([df1,df2,df3])

df1 = pd.read_csv(os.path.join('open_images', 'train-images-boxable-with-rotation.csv'))
df2 = pd.read_csv(os.path.join('open_images', 'validation-images-with-rotation.csv'))
df3 = pd.read_csv(os.path.join('open_images', 'test-images-with-rotation.csv'))
meta = pd.concat([df1,df2,df3])

data.set_index('LabelName', inplace=True)
all_ids = data.ImageID.unique()
meta.set_index('ImageID', inplace=True)
meta = meta[meta.index.isin(all_ids)]
cats = data.index.unique()
objects = {}
mid2freq = {}
for cat in cats:
    rows = data.loc[[cat]]
    ids = rows.loc[:,'ImageID'].values
    rot = meta.loc[ids, 'Rotation'].values
    box = rows.loc[:,'XMin':'YMax'].values #xmin, xmax, ymin, ymax
    box = np.asarray([fix_order(rotate(b,r)) for b,r in zip(box,rot)])
    objects[cat] = (box, ids)
    mid2freq[cat] = len(rows)
joblib.dump(objects,os.path.join('data','data.joblib'))

id2url = dict(zip(meta.index, meta.OriginalURL))
joblib.dump(id2url,os.path.join('data','id2url.joblib'))

id2rot = dict(zip(meta.index, meta.Rotation))
joblib.dump(id2rot,os.path.join('data','id2rot.joblib'))

id2set = dict(zip(meta.index, meta.Subset))
joblib.dump(id2set,os.path.join('data','id2set.joblib'))

desc = pd.read_csv(os.path.join('open_images', 'class-descriptions-boxable.csv'), names=['mid','label'])
mid2label=dict(zip(desc.mid,desc.label))
joblib.dump(mid2label,os.path.join('data','mid2label.joblib'))
label2mid=dict(zip(desc.label,desc.mid))
joblib.dump(label2mid,os.path.join('data','label2mid.joblib'))

norm = sum(mid2freq.values())
mid2freq = {k:v/norm for k,v in mid2freq.items()}
joblib.dump(mid2freq,os.path.join('data','mid2freq.joblib'))
pd.DataFrame((mid2label[k],v) for k,v in mid2freq.items()).sort_values(1,0).to_csv(os.path.join('data','freq.csv'), index=False, header=False)
mid2rank = {k:1/np.sqrt(i+1) for i,(k,v) in enumerate(sorted(mid2freq.items(), key=lambda x: x[::-1]))}
joblib.dump(mid2rank,os.path.join('data','mid2rank.joblib'))

with open(os.path.join('open_images', 'bbox_labels_600_hierarchy.json')) as f:
    hierarchy = json.load(f)

def find_parents(tree, mid, parent=None, ptype=None, level=-1):
    parents = []
    if level==-1:
        tree = [tree]
    for item in tree:
        if item['LabelName'] == mid:
            parents.append((parent, ptype, level))
        if 'Subcategory' in item:
            parents.extend(find_parents(item['Subcategory'], mid, item['LabelName'], 'Subcategory', level+1))
        if 'Part' in item:
            parents.extend(find_parents(item['Part'], mid, item['LabelName'], 'Part', level+1))
    if level==-1:
        filtered = []
        for parent in sorted(parents, reverse=True):
            if len(filtered)==0 or parent[:2] != filtered[-1][:2]:
                filtered.append(parent)
        parents = filtered
    return parents

mid2parents = {mid: find_parents(hierarchy, mid) for mid in mid2label}
joblib.dump(mid2parents,os.path.join('data','mid2parents.joblib'))

def find_all_children(mid2parents):
    children = defaultdict(list)
    for child, parents in mid2parents.items():
        for parent in parents:
            children[parent[0]].append((child, parent[1], parent[2]+1))
    return dict(children)

mid2children = find_all_children(mid2parents)
joblib.dump(mid2children,os.path.join('data','mid2children.joblib'))

