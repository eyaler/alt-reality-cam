import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageColor, ImageFont, ImageFile
from urllib.request import urlopen, Request
from io import BytesIO
import joblib
from open_images_starter import text, visual
from open_images_starter.region import Region
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

id2rot = joblib.load(os.path.join('data','id2rot.joblib'))
id2set = joblib.load(os.path.join('data','id2set.joblib'))
mid2label = joblib.load(os.path.join('data','mid2label.joblib'))

def id2url(index):
    return 'https://s3.amazonaws.com/open-images-dataset/' + id2set[index] + '/' + index + '.jpg'

def get_image_from_s3(index, save_path=None, show=False):
    image = get_image(id2url(index), rotate=id2rot[index])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image.save(save_path, format="JPEG", quality=90)

    if show:
        display_image(image)

    return image


def get_image(url, rotate='auto'):
    request = Request(url, headers={'User-Agent': "Magic Browser"})
    response = urlopen(request, timeout=30)
    image_data = response.read()
    image_data = BytesIO(image_data)
    pil_image = Image.open(image_data)

    if rotate=='auto':
        rotate = None
        try:
            exif = pil_image._getexif()
            if exif[274] == 3:
                rotate = 180
            elif exif[274] == 6:
                rotate = 270
            elif exif[274] == 8:
                rotate = 90
        except:
            pass

    rotate = np.nan_to_num(rotate)
    if rotate:
        pil_image = pil_image.rotate(rotate, expand=True)

    return pil_image.convert('RGBA').convert('RGB')


def display_image(image):
    plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)
    plt.draw()
    while not plt.waitforbuttonpress(0): pass
    plt.close()

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
    """Adds a bounding box to an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
               (left, top)],
              width=thickness,
              fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                       fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill="black",
                  font=font)
        text_bottom -= text_height - 2 * margin


def get_image_boxes(objects, index):
    result = {"detection_boxes": [], "detection_class_names": []}
    for cat, obj in objects.items():
        for box, id in zip(*obj): #box = ymin, xmin, ymax, xmax
            if id == index:
                result["detection_boxes"].append(box)
                result["detection_class_names"].append(cat)
    return result


def draw_boxes(image, boxes, class_names, scores=None, max_boxes=None, min_score=None, uid=None, style='new', save_path=None, show=False, show_size=False):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    used_classes = set()
    used_uid = set()
    inds = []
    for i in range(len(boxes)):
        if (min_score is None or scores is None or scores[i] >= min_score) and (uid is None or uid[i] not in used_uid):
            inds.append(i)
            used_classes.add(class_names[i])
            if uid is not None:
                used_uid.add(uid[i])
            if max_boxes is not None and len(inds)>=max_boxes:
                break

    if style == 'new':
        colors = visual.generate_colors(len(used_classes), saturation=0.8, hue_offset=0.35, hue_range=0.5)
        color_map = dict(zip(used_classes, colors))
    else:
        colors = list(ImageColor.colormap.values())

    font = ImageFont.load_default()
    image = np.asarray(image).copy()
    for i in inds:
        ymin, xmin, ymax, xmax = tuple(boxes[i].tolist())
        class_name = class_names[i]
        display_str = mid2label[class_name]
        if scores is not None:
            display_str = "{}: {}%".format(display_str, int(100 * scores[i]))
        if show_size:
            display_str = "{} ({:.4f})".format(display_str, (xmax-xmin)*(ymax-ymin))
        if style=='new':
            im_height, im_width = image.shape[:2]
            region = Region(xmin*im_width,xmax*im_width,ymin*im_height,ymax*im_height)
            image = visual.draw_regions(image, [region], color=(0, 0, 0), thickness=10, strength=0.3)
            image = visual.draw_regions(image, [region], color=color_map[class_name], thickness=4, overlay=True)

            image = text.label_region(image, display_str, region, color=color_map[class_name],
                                      bg_opacity=0.7, overlay=True, font_size=20, inside=region.top <= 30)

        else:

            color = colors[hash(class_name) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(image).save(save_path, format="JPEG", quality=90)

    if show:
        display_image(image)

    return image
