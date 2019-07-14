import os
import tensorflow as tf
import tensorflow_hub as hub
import tempfile
from PIL import Image, ImageOps
from util import display_image, get_image
from time import time

def download_and_resize_image(url, new_width=512, new_height=512, save_path=None,
                              show=False):
    if save_path is None:
        _, save_path = tempfile.mkstemp(suffix=".jpg")
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pil_image = get_image(url, rotate='auto')
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    pil_image_rgb.save(save_path, format="JPEG", quality=90)
    if show:
        display_image(pil_image)
    return save_path

global session, result, decoded_image
print("The following GPU devices are available: %s" % tf.test.gpu_device_name())
module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

with tf.Graph().as_default():
    detector = hub.Module(module_handle)

    image_string_placeholder = tf.placeholder(tf.string)
    decoded_image = tf.image.decode_jpeg(image_string_placeholder)
    # Module accepts as input tensors of shape [1, height, width, 3], i.e. batch
    # of size 1 and type tf.float32.
    decoded_image_float = tf.image.convert_image_dtype(
        image=decoded_image, dtype=tf.float32)
    module_input = tf.expand_dims(decoded_image_float, 0)
    result = detector(module_input, as_dict=True)
    init_ops = [tf.global_variables_initializer(), tf.tables_initializer()]
    session = tf.Session()
    session.run(init_ops)

def process_image(image_url, res_x=512, res_y=512, save_path=None, show=False):
    downloaded_image_path = download_and_resize_image(image_url, new_width=res_x, new_height=res_y, save_path=save_path, show=show)

    # Load the downloaded and resized image and feed into the graph.
    with tf.gfile.Open(downloaded_image_path, "rb") as binfile:
        image_string = binfile.read()

    start = time()
    result_out, image_out = session.run([result, decoded_image], feed_dict={image_string_placeholder: image_string})
    print("Found %d objects. took %s sec" % (len(result_out["detection_scores"]), time() - start))
    result_out["detection_class_names"] = [name.decode('ascii') for name in result_out["detection_class_names"]]

    return image_out, result_out
