from process_image import process_image
from util import draw_boxes, get_image_from_s3, id2url
import os
import re
import glob
import json
import joblib
from datetime import datetime
from match import find_similar, box_area
import pandas as pd
from time import time, sleep
from urllib.error import URLError
from socket import gaierror, error, timeout
import shutil
from build_biases import get_manual_bias_labels
from itertools import islice
import numpy as np

suppression_threshold = 0.3
suppression_topk = 5
min_area = 0.009
images_topk = 5
biases_topk = 3
hierarchy_factor = 0.5
polypedo_discount = 1
res_x = 1920
res_y = 1080
serve_path = 'serve'
show_image = False
show_overlay = False
save_output_overlay = True
test_mode = False
wait = 20
twitter_poll_wait = 20
twitter_error_wait = 1000
show_size = True
suppress_duplicate_matches = True
required_bias_objects_types = 1
ignore_bias_objects_types = ['Human face','Man','Clothing','Person']
override_start_folder = None

if os.path.exists('serve_path.txt'):
    with open('serve_path.txt') as f:
        serve_path = f.read()

mid2label=joblib.load(os.path.join('data','mid2label.joblib'))
label2mid=joblib.load(os.path.join('data','label2mid.joblib'))
mid2subcnt = joblib.load(os.path.join('data','mid2subcnt.joblib'))
id2mids=joblib.load(os.path.join('data','id2mids.joblib'))
bias_map = pd.read_csv('biases.csv', index_col='zero')

biases = ['zero', 'gender', 'war', 'money', 'love', 'fear']
input_loc = 'image_urls.txt'
get_twitter = False
ret_twitter = True
pt_msg = 'see more at the Deep Feeling exhibition @PT_Museum http://www.petachtikvamuseum.com/he/Exhibitions.aspx?eid=4987'

assert {label for sublist in bias_map.values for labels in sublist for label in labels.split('/')} <= set(label2mid)
def get_biases(label, bias):
    if bias == bias_map.index.name or label not in bias_map.index:
        return label
    result = bias_map.loc[label,bias]
    return result if len(result)>0 else label

bias_labels = get_manual_bias_labels(biases)
def bias_objects_types(labels, bias):
    return sum(mid2label[label] in bias_labels[bias] for label in set(labels) if mid2label[label] not in ignore_bias_objects_types)


once = True
wait_notice = False
while once or get_twitter:
    if get_twitter:
        if not wait_notice:
            print('twitter server mode')
            wait_notice = True
        input_loc = []
        timestamps = []
        twitter_user = []
        twitter_bias = []
        status_id = []
        import twitter
        with open('twitter_key.txt') as fin:
            lines = fin.read().splitlines()
        api = twitter.Api(*lines)
        last_id = None
        if os.path.exists(os.path.join(serve_path,'twitter_id.txt')):
            with open(os.path.join(serve_path,'twitter_id.txt')) as fin:
                last_id = int(fin.read())
        try:
            mentions = api.GetMentions(since_id=last_id)
        except Exception as e:
            print(e)
            print('Error getting tweets')
            sleep(twitter_error_wait)
            continue
        for mention in reversed(mentions):
            media = mention.media
            if media:
                input_loc.append(media[0].media_url)
                timestamps.append(datetime.fromtimestamp(mention.created_at_in_seconds).isoformat(' '))
                twitter_user.append(mention.user.screen_name)
                twitter_bias.append(re.findall(r'#(\w+)', mention.text))
                status_id.append(mention.id)
        if not input_loc:
            sleep(twitter_poll_wait)
            continue
    elif type(input_loc) == str:
        once = False
        if os.path.isdir(input_loc):
            input_loc = sorted('file:///'+f for f in glob.iglob(os.path.join(os.path.abspath(input_loc),'**','*'), recursive=True) if os.path.isfile(f) and os.path.splitext(f)[1]!='.ini')
        elif os.path.splitext(input_loc)[1] == '.txt':
            with open(input_loc) as fin:
                input_loc = [line for line in fin.read().splitlines() if line.strip()]
        else:
            input_loc = ['file:///'+os.path.abspath(input_loc)]

    counter = 0
    if test_mode:
        print('test mode will save to serve folder 0')
    elif override_start_folder is not None:
        counter = override_start_folder
        print('will start serve folders at %d (override)' % counter)
    else:
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
        print('will start serve folders at %d' % counter)

    for cnt, image_url in enumerate(input_loc):
        image_url = image_url.strip()
        print('\nProcessing: %s'%image_url)
        print('\n(additional images pending: %d)' % len(input_loc)-cnt-1)
        folder = os.path.join(serve_path, str(counter))
        result = None
        while result is None:
            os.makedirs(folder)
            try:
                start = time()
                image, result = process_image(image_url, res_x=res_x, res_y=res_y, save_path=os.path.join(folder, 'input.jpg'), show=show_image)
            except (URLError, gaierror, error, timeout, ConnectionError) as e:
                print(e)
                if image_url.startswith('file:'):
                    print('Error processing file - skipping')
                    break
                print('Error retrieving URL - will retry in %d sec'%wait)
                os.rmdir(folder)
                sleep(wait)
                continue
            except Exception as e:
                print(e)
                print('Error processing file - skipping')
                break
        if result is None:
            shutil.rmtree(folder)
            continue

        results = list(zip(*[x for x in sorted(zip(result["detection_scores"], result["detection_boxes"], result["detection_class_names"]), reverse=True) if x[0] > suppression_threshold and box_area(x[1])>=min_area][:suppression_topk]))
        if len(results)==0:
            shutil.rmtree(folder)
            print('No detected objects found above thresholds - skipping')
            continue
        result["detection_scores"], result["detection_boxes"], result["detection_class_names"] = results
        image_with_boxes = draw_boxes(image, result["detection_boxes"], result["detection_class_names"],
                                      scores=result["detection_scores"], style='new',
                                      save_path=os.path.join(folder, 'input_overlay.jpg'), show=show_overlay, show_size=show_size)
        if get_twitter:
            timestamp = timestamps[cnt]
        else:
            timestamp = datetime.utcnow().isoformat(' ')
        height, width = image.shape[:2]
        input_list = [{'bbox': [int(box[1] * width), int(box[0] * height), int(box[3] * width), int(box[2] * height)],
                       'label': mid2label[label], 'score': score.item()} for box, label, score in
                      zip(result["detection_boxes"], result["detection_class_names"], result["detection_scores"])]
        input_query = [mid2label[label] for label in result["detection_class_names"]]
        output_list = []

        i = 0
        seen_images = set()
        have_result = False
        for bias in biases:
            query_names = [get_biases(mid2label[label], bias).split('/')[:biases_topk] for label in result["detection_class_names"]]
            query_labels = [[label2mid[name] for name in names] for names in query_names]
            sim = find_similar(result["detection_boxes"], query_labels, hierarchy_factor=hierarchy_factor, polypedo_discount=polypedo_discount, min_area=min_area, allowed_labels=None if bias=='zero' else bias_labels[bias])
            sim = (s for s in sim if (not suppress_duplicate_matches or s[0] not in seen_images) and (not required_bias_objects_types or bias == 'zero' or bias_objects_types(s[5],bias)>=required_bias_objects_types))
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
            anchor = []
            transform = []
            imgs = []
            for k, s in enumerate(sim):
                if bias!='zero':
                    seen_images.add(s[0])
                have_s3 = None
                while have_s3 is None:
                    try:
                        img = get_image_from_s3(s[0], save_path=os.path.join(folder, 'bias%d_img%d.jpg' % (i,k+1)), show=show_image)
                        have_s3 = True
                    except (URLError, gaierror, error, timeout, ConnectionError) as e:
                        print(e)
                        print('Error retrieving from S3 - will retry in %d sec' % wait)
                        sleep(wait)
                        continue
                scores.append(s[9])

                if save_output_overlay or show_overlay:
                    draw_boxes(img, s[2], s[4], uid=s[3], save_path=os.path.join(folder, 'bias%d_img%d_overlay.jpg' % (i,k+1)) if save_output_overlay else None, show=show_overlay, show_size=show_size)

                bias_other = id2mids[s[0]].copy()
                for label in s[5]:
                    bias_other.remove(label)
                found.append([mid2label[label] for label in s[5]])
                other.append([mid2label[label] for label in bias_other])
                anchor.append([mid2label[label] for label in s[7]])
                transform.append([(input_query[matched_ind],mid2label[label]) for matched_ind,label in zip(s[1],s[4])])
                imgs.append(id2url(s[0]))

            output_list.append({'type':bias, 'query':query_names, 'found':found, 'other':other, 'anchor':anchor, 'transform':transform, 'scores':scores, 'count':len(sim), 'url':imgs})
            i += 1

        if i==0:
            shutil.rmtree(folder)
            continue

        json_dict = {'index': counter, 'time':timestamp, 'input_url':image_url, 'input':input_list, 'query':input_query, 'output':output_list}
        for json_folder in [serve_path, folder]:
            with open(os.path.join(json_folder, 'labels.json'), 'w') as fout:
                json.dump(json_dict, fout, indent='\t')

        if get_twitter:
            if ret_twitter:
                have_biases = {output['type']: ind for ind,output in enumerate(output_list)}
                if twitter_bias[cnt] and twitter_bias[cnt][0].lower() in have_biases:
                    ret_bias = twitter_bias[cnt][0].lower()
                else:
                    ret_bias = np.random.choice([b for b in have_biases if b!='zero'])
                output = output_list[have_biases[ret_bias]]
                ret_img = output['url'][np.random.randint(output['count'])]
                try:
                    api.PostUpdate('@'+twitter_user[cnt]+' #'+ret_bias+' '+pt_msg, in_reply_to_status_id=status_id[cnt], media=ret_img)
                except Exception as e:
                    print(e)
                    print('Error posting twitter reply')
            with open(os.path.join(serve_path,'twitter_id.txt'), 'w') as fout:
                fout.write(str(status_id[cnt]))

        counter += 1
        wait_notice = True
        print('total: %.1f min'%((time()-start)/60))
        if test_mode:
            break

    print('\nDone.\n')