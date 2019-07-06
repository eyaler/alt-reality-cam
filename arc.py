from process_image import process_image
from util import draw_boxes, get_image_from_s3
import os
import json
import joblib
from datetime import datetime
from match import find_similar, box_area
import pandas as pd
from time import time, sleep
from urllib.error import URLError
from socket import gaierror
import shutil
from build_biases import get_manual_bias_labels
from itertools import islice

suppression_threshold = 0.3
suppression_topk = 5
min_area = 0.009
images_topk = 5
hierarchy_factor = 0.5
polypedo_discount = 1
res_x = 1920
res_y = 1080
serve_path = 'serve'
show_image = False
show_overlay = False
save_output_overlay = True
test_mode = False
wait = 5
show_size = True
suppress_duplicate_matches = True
required_bias_objects = 1

mid2label=joblib.load(os.path.join('data','mid2label.joblib'))
label2mid=joblib.load(os.path.join('data','label2mid.joblib'))
mid2subcnt = joblib.load(os.path.join('data','mid2subcnt.joblib'))
id2mids=joblib.load(os.path.join('data','id2mids.joblib'))
bias_map = pd.read_csv('biases.csv', index_col='zero')

biases = ['zero', 'gender', 'military', 'money', 'love', 'fear']
input_loc = 'image_urls.txt'
#input_loc = 'nohar'

assert {label for sublist in bias_map.values for label in sublist} <= set(label2mid)
def get_bias(label, bias):
    if bias == bias_map.index.name or label not in bias_map.index:
        return label
    result = bias_map.loc[label,bias]
    return result if len(result)>0 else label

def get_biases(labels, bias):
    return [label2mid[get_bias(mid2label[l], bias)] for l in labels]

bias_labels = get_manual_bias_labels()
def bias_objects(labels, bias):
    return sum(mid2label[label] in bias_labels[bias] for label in labels)

counter = 0
if not test_mode:
    os.makedirs(serve_path, exist_ok=True)
    for folder in os.listdir(serve_path):
        if not os.path.isdir(os.path.join(serve_path, folder)):
            continue
        try:
            folder = int(folder)
        except:
            continue
        counter = max(counter, folder)
    counter += 1
    print('starting serve folders at %d'%counter)
else:
    print('test mode will save to serve folder 0')

if type(input_loc) == str:
    if os.path.isdir(input_loc):
        input_loc = ['file:///'+os.path.abspath(os.path.join(input_loc,f)) for f in os.listdir(input_loc)]
    elif input_loc.endswith('.txt'):
        with open(input_loc) as fin:
            input_loc = fin.read().splitlines()
    else:
        input_loc = ['file:///'+os.path.abspath(input_loc)]

for image_url in input_loc:
    start = time()
    print('\nProcessing: '+image_url.strip())

    folder = os.path.join(serve_path, str(counter))
    os.makedirs(folder, exist_ok=True)
    result = None
    while result is None:
        try:
            start = time()
            image, result = process_image(image_url, res_x=res_x, res_y=res_y, save_path=os.path.join(folder, 'input.jpg'), show=show_image)
        except (URLError, gaierror) as e:
            print(e)
            print('Error retrieving URL - will retry in %d sec'%wait)
            sleep(wait)
            continue
        except Exception as e:
            print(e)
            print('Error processing file - skipping')
            break
    if result is None:
        shutil.rmtree(folder)
        continue

    detection_cnt_subs = [mid2subcnt[mid] for mid in result["detection_class_names"]]

    results = list(zip(*[x for x in sorted(zip(detection_cnt_subs, result["detection_scores"], result["detection_boxes"], result["detection_class_names"]), key=lambda x: (x[0], -x[1], x[2], x[3])) if x[1] > suppression_threshold and box_area(x[2])>=min_area][:suppression_topk]))
    if len(results)==0:
        shutil.rmtree(folder)
        print('no detected objects found above thresholds - skipping')
        continue
    _, result["detection_scores"], result["detection_boxes"], result["detection_class_names"] = results
    image_with_boxes = draw_boxes(image, result["detection_boxes"], result["detection_class_names"],
                                  scores=result["detection_scores"], style='new',
                                  save_path=os.path.join(folder, 'input_overlay.jpg'), show=show_overlay, show_size=show_size)
    timestamp = datetime.now().isoformat(' ')
    height, width = image.shape[:2]
    input_list = [{'bbox': [int(box[1] * width), int(box[0] * height), int(box[3] * width), int(box[2] * height)],
                   'label': mid2label[label], 'score': score.item()} for box, label, score in
                  zip(result["detection_boxes"], result["detection_class_names"], result["detection_scores"])]
    output_list = []

    i = 0
    seen_images = set()
    have_result = False
    for bias in biases:
        biased_class_names = get_biases(result["detection_class_names"], bias)
        #if bias!='zero' and biased_class_names == list(result["detection_class_names"]):
        #    print('%s: no bias shift found - skipping'%bias)
        #    continue

        sim = find_similar(result["detection_boxes"], biased_class_names, hierarchy_factor=hierarchy_factor, polypedo_discount=polypedo_discount, min_area=min_area)
        sim = (s for s in sim if (not suppress_duplicate_matches or s[0] not in seen_images) and (not required_bias_objects or bias == 'zero' or bias_objects(s[5],bias)>=required_bias_objects))
        sim = list(islice(sim, images_topk))
        if len(sim)==0:
            print('%s: no matches found'%bias)
            continue

        if not have_result:
            have_result = True
            print('folder=%d' % counter)

        scores = []
        found = []
        other = []
        for k, s in enumerate(sim):
            if bias!='zero':
                seen_images.add(s[0])
            have_s3 = None
            while have_s3 is None:
                try:
                    img = get_image_from_s3(s[0], save_path=os.path.join(folder, 'bias%d_img%d.jpg' % (i,k+1)), show=show_image)
                    have_s3 = True
                except (URLError, gaierror) as e:
                    print(e)
                    print('Error retrieving from S3 - will retry in %d sec' % wait)
                    sleep(wait)
                    continue
            scores.append(s[7])

            if save_output_overlay or show_overlay:
                draw_boxes(img, s[2], s[4], uid=s[3], save_path=os.path.join(folder, 'bias%d_img%d_overlay.jpg' % (i,k+1)) if save_output_overlay else None, show=show_overlay, show_size=show_size)

            bias_other = id2mids[s[0]].copy()
            for label in s[5]:
                bias_other.remove(label)
            found.append([mid2label[l] for l in s[5]])
            other.append([mid2label[l] for l in bias_other])

        query = [mid2label[label] for label in biased_class_names]
        output_list.append({'type':bias, 'query':query, 'found':found, 'other':other, 'scores':scores, 'count':len(sim)})
        i += 1

    if i==0:
        shutil.rmtree(folder)
        continue

    json_dict = {'index': counter, 'time':timestamp, 'input':input_list, 'output':output_list}
    for json_folder in [serve_path, folder]:
        with open(os.path.join(json_folder, 'labels.json'), 'w') as fout:
            json.dump(json_dict, fout, indent='\t')

    counter += 1
    print('total: %.1f min'%((time()-start)/60))
    if test_mode:
        break

print('\nDone.')