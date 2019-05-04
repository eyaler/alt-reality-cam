from process_image import process_image
from util import draw_boxes, get_image_from_s3
import os
import json
import joblib
from datetime import datetime
from match import find_similar, filter_duplicate_boxes

supression_threshold = 0.3
supression_topk = 5
images_topk = 5
hierarchy_factor = 0.5
res_x = 1920
res_y = 1080
serve_path = 'serve'
show_image = False
show_overlay = True
save_output_overlay = True
test_mode = False

mid2label=joblib.load(os.path.join('data','mid2label.joblib'))

counter = 0
if not test_mode:
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

with open('image_urls.txt') as fin:
    for image_url in fin:
        print('\nProcessing: '+image_url.strip())

        folder = os.path.join(serve_path, str(counter))
        os.makedirs(folder, exist_ok=True)
        image, result = process_image(image_url, res_x=res_x, res_y=res_y, save_path=os.path.join(folder, 'input.jpg'), show=show_image)
        result["detection_scores"], result["detection_boxes"], result["detection_class_names"] = zip(*[x for x in sorted(zip(result["detection_scores"], result["detection_boxes"], result["detection_class_names"]), reverse=True) if x[0]>supression_threshold][:supression_topk])
        image_with_boxes = draw_boxes(image, result["detection_boxes"], result["detection_class_names"], scores=result["detection_scores"], style='new', save_path=os.path.join(folder, 'input_overlay.jpg'), show=show_overlay)

        timestamp = datetime.now().isoformat(' ')

        height, width = image.shape[:2]
        input_list = [{'bbox':[int(box[1]*width),int(box[0]*height),int(box[3]*width),int(box[2]*height)], 'label':mid2label[label], 'score':score.item()} for box,label,score in zip (result["detection_boxes"], result["detection_class_names"], result["detection_scores"])]
        output_list = []
        query = [mid2label[label] for label in result["detection_class_names"]]
        sim = find_similar(result["detection_boxes"], result["detection_class_names"], top_k=images_topk, hierarchy_factor=hierarchy_factor)
        scores = []
        for k in range(len(sim)):
            img = get_image_from_s3(sim[k][0], save_path=os.path.join(folder, 'bias0_img%d.jpg' % (k+1)), show=show_image)
            scores.append(sim[k][6])
            if save_output_overlay or show_overlay:
                draw_boxes(img, sim[k][2], sim[k][4], uid=sim[k][3], save_path=os.path.join(folder, 'bias0_img%d_overlay.jpg' % (k+1)) if save_output_overlay else None, show=show_overlay)
        found = [mid2label[l] for s in sim for l in filter_duplicate_boxes(s[3], s[4])]
        output_list.append({'type':'zero', 'query':query, 'found':found, 'scores':scores, 'count':len(sim)})
        json_dict = {'index': counter, 'time':timestamp, 'input':input_list, 'output':output_list}
        for json_folder in [serve_path, folder]:
            with open(os.path.join(json_folder, 'labels.json'), 'w') as fout:
                json.dump(json_dict, fout)

        counter += 1
        if test_mode:
            break

print('\nDone.')