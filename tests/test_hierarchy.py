import json
import joblib
import os

with open(os.path.join('open_images_v4', 'bbox_labels_600_hierarchy.json')) as f:
    hierarchy = json.load(f)

mid2label = joblib.load(os.path.join('data','mid2label.joblib'))
mid2parents = joblib.load(os.path.join('data','mid2parents.joblib'))

lens = set()
for mid,label in mid2label.items():
    parents = mid2parents[mid]
    pretty_parents = str(['%s (%s)' % (mid2label[parent[0]], parent) if parent[0] in mid2label else parent for parent in parents])
    print('%s (%s): %s cnt=%d' % (label, mid, pretty_parents, len(parents)))
    lens.add(len(parents))
print(lens)

