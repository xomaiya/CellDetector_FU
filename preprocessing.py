from bboxes import convert_to_xywh, swap_xy
import tensorflow as tf


def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance

    :param image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
    :param boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    :return: randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes


def preprocess_data(sample):
    """Applies preprocessing step to a single sample

    :param sample: A dict representing a single training sample.

    :return image: Resized and padded image with random horizontal flipping applied.
    :return bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[y, x, height, weight]`.
    :return class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    image = sample["image"]
    bbox = sample["objects"]["bbox"]
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

    image, bbox = random_flip_horizontal(image, bbox)
    bbox = swap_xy(bbox)

    image_shape = (256, 256)
    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[0],
            bbox[:, 1] * image_shape[1],
            bbox[:, 2] * image_shape[0],
            bbox[:, 3] * image_shape[1],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id
