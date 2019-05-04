import joblib
import os

objects = joblib.load(os.path.join('data','data.joblib'))
mid2parents = joblib.load(os.path.join('data','mid2parents.joblib'))
mid2children = joblib.load(os.path.join('data','mid2children.joblib'))
mid2label = joblib.load(os.path.join('data','mid2label.joblib'))

ids = set()
for parents in mid2parents.values():
    ids.update([(id,level) for id,ptype,level in parents])

ids = sorted(ids, key=lambda x: -x[1])
print([(mid2label[id] if id in mid2label else id, level, id in objects) for id,level in ids])

for child,parents in mid2parents.items():
    for parent in parents:
        if parent[0] == '/m/0bl9f':
            print(mid2label[child])

for parent,children in sorted(mid2children.items(), key=lambda x: sum([child[1]=='Subcategory' for child in x[1]])):
    print(sum([child[1]=='Subcategory' for child in children]),mid2label[parent] if parent in mid2label else parent,[(mid2label[child[0]], child[1], child[2]) for child in children if child[1]=='Subcategory'])