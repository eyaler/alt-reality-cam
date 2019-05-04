import pandas as pd
import joblib
from util import draw_boxes, display_image, get_image_from_s3, get_image_boxes
import os

id = 'bf55524580ebef48'

df1 = pd.read_csv(os.path.join('open_images_v4', 'train-annotations-bbox.csv'))
df2 = pd.read_csv(os.path.join('open_images_v4', 'validation-annotations-bbox.csv'))
df3 = pd.read_csv(os.path.join('open_images_v4', 'test-annotations-bbox.csv'))
data = pd.concat([df1,df2,df3])

mid2label = joblib.load(os.path.join('data','mid2label.joblib'))
filtered = data[data.ImageID==id]
labels = filtered.LabelName
boxes = filtered.XMin
print([(mid2label[l],b) for l,b in zip(labels,boxes)])

image = get_image_from_s3(id)

objects = joblib.load(os.path.join('data','data.joblib'))

result = get_image_boxes(objects, id)
print(result)

image_with_boxes = draw_boxes(image, result["detection_boxes"], result["detection_class_names"])
display_image(image_with_boxes)
