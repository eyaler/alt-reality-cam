import joblib
import os

objects = joblib.load(os.path.join('..','data','data.joblib'))
mid2parents = joblib.load(os.path.join('..','data','mid2parents.joblib'))
mid2children = joblib.load(os.path.join('..','data','mid2children.joblib'))
mid2label = joblib.load(os.path.join('..','data','mid2label.joblib'))

ids = set()
for parents in mid2parents.values():
    ids.update([(id,level) for id,ptype,level in parents])

ids = sorted(ids, key=lambda x: -x[1])
print([(mid2label[id] if id in mid2label else id, level, id in objects) for id,level in ids])

for child,parents in mid2parents.items():
    for parent in parents:
        if parent[0] == '/m/0bl9f':
            print(mid2label[child])

def count_subs(mid):
    if mid not in mid2children:
        return 0
    return sum(child[1] == 'Subcategory' for child in mid2children[mid])

def count_subs_final(mid):
    if mid not in mid2children:
        return 0
    return sum(count_subs_final(child[0]) if child[0] in mid2children else 1 for child in mid2children[mid] if child[1] == 'Subcategory')

def count_subs_multi(mid):
    if mid not in mid2children:
        return 0
    return sum(count_subs_multi(child[0])+1 if child[0] in mid2children else 1 for child in mid2children[mid] if child[1] == 'Subcategory')

def find_subs_final(mid):
    if mid not in mid2children:
        return []
    return [item for sublist in [find_subs_final(child[0]) if child[0] in mid2children else [child[0]] for child in mid2children[mid] if child[1] == 'Subcategory'] for item in sublist]

def find_subs_multi(mid):
    if mid not in mid2children:
        return []
    return [item for sublist in [find_subs_multi(child[0])+[child[0]] if child[0] in mid2children else [child[0]] for child in mid2children[mid] if child[1] == 'Subcategory'] for item in sublist]

for mid in mid2label:
    assert count_subs_final(mid) == len(find_subs_final(mid))
    assert count_subs_multi(mid) == len(find_subs_multi(mid))

for parent,children in sorted(mid2children.items(), key=lambda x: (count_subs(x[0]), count_subs_multi(x[0]))):
    print(count_subs(parent),len(set(find_subs_final(parent))), count_subs_final(parent),len(set(find_subs_multi(parent))), count_subs_multi(parent),mid2label[parent] if parent in mid2label else parent,[(mid2label[child[0]], child[1], child[2]) for child in children if child[1]=='Subcategory'])

mid2freq = joblib.load(os.path.join('..','data','mid2freq.joblib'))
mid2hfreq = joblib.load(os.path.join('..','data','mid2hfreq.joblib'))
label2mid = joblib.load(os.path.join('..','data','label2mid.joblib'))
id = label2mid['Food']
children = [[id]]
i = 0
while True:
    print(i, sum(mid2freq[child] if child in mid2freq else 0 for child in set(item for sublist in children for item in sublist)))
    i += 1
    new_children = [x[0] for child in children[i-1] if child in mid2children for x in mid2children[child]]
    if not new_children:
        break
    children.append(new_children)
print('h',mid2hfreq[id])
