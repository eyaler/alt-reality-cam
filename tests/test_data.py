import joblib
import pandas as pd
from util import draw_boxes, display_image, get_image_from_s3, get_image_boxes
import os

objects = joblib.load(os.path.join('data','data.joblib'))
id2url = joblib.load(os.path.join('data','id2url.joblib'))

meta = pd.read_csv(os.path.join('open_images_v4', 'validation-images-with-rotation.csv'))
meta.set_index('ImageID', inplace=True)

print('data loaded')

for index, row in meta.iterrows():
    if index not in id2url:
        continue

    #if not np.isnan(row.Rotation):
    if row.Rotation != 270:
        continue

    '''
    image_url = id2url[index]
    try:
        image = get_image(image_url, rotate=id2rot[index])
    except:
        print('error downloading: ' + image_url)
        continue
    '''

    image = get_image_from_s3(index)

    result = get_image_boxes(objects, index)
    print(row.Rotation)
    print(result)

    image_with_boxes = draw_boxes(image, result["detection_boxes"], result["detection_class_names"])
    display_image(image_with_boxes)
