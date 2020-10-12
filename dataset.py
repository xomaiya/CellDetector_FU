import tensorflow as tf
from data_loader import SlideContainer
import random
import numpy as np
from typing import List
from tqdm import tqdm


class MyDataset():
    """
    Dataset class with two generators: finite (for validation) and infinite (for training)

    :param slides: list(SlideContainer) as return of get_slides()
    :param bboxes: list of bounding boxes
    """

    def __init__(self, slides: List[SlideContainer], bboxes):
        self.slides = slides
        self.bboxes = bboxes
        self.subimgs = self.get_subimgs()

        self.bb_arr = [np.array([i[0] for i in bboxes_sl]) for bboxes_sl in self.bboxes]
        self.lbl_arr = [np.array([i[1] for i in bboxes_sl]) for bboxes_sl in self.bboxes]

        self.dataset = tf.data.Dataset.from_generator(self.generator,
                                                      output_types={'image': tf.float32,
                                                                    'objects': {'bbox': tf.float32, 'label': tf.int32}},
                                                      output_shapes={'image': tf.TensorShape([256, 256, 3]),
                                                                     'objects': {'bbox': tf.TensorShape([None, 4]),
                                                                                 'label': tf.TensorShape([None])}}
                                                      )

        self.inf_dataset = tf.data.Dataset.from_generator(self.infinite_generator,
                                                          output_types={'image': tf.float32,
                                                                        'objects': {'bbox': tf.float32,
                                                                                    'label': tf.int32}},
                                                          output_shapes={'image': tf.TensorShape([256, 256, 3]),
                                                                         'objects': {'bbox': tf.TensorShape([None, 4]),
                                                                                     'label': tf.TensorShape([None])}}
                                                          )

    def get_random_crop_around(self, slide, bbox):
        """
        Get coordinates of random patch from slide, that contains bbox. 
        Size of the patch defines as (slide.width, slide.height).

        :param slide: big image (slide.slide.dimensions)
        :param bbox: some bounding box for the slide

        :return: tuple of coordinates of new patch
        """

        xl = random.randint(np.clip(bbox[2] - slide.width, 0, slide.slide.dimensions[0]),
                            np.clip(bbox[0], 0, slide.slide.dimensions[0] - slide.width))
        yl = random.randint(np.clip(bbox[3] - slide.height, 0, slide.slide.dimensions[1]),
                            np.clip(bbox[1], 0, slide.slide.dimensions[1] - slide.height))
        xr = xl + slide.width
        yr = yl + slide.height
        return (xl, yl, xr, yr)

    def get_all_landed_bboxes(self, subimg, bb_arr, lbl_arr):
        """
        Get all the bounding boxes for sub-image.

        :param subimg: patch from the slide
        :param bboxes: bounding boxes for the slide

        :return: bounding boxes (list) and corresponding labels (another list) that in the subimg
        """

        condition = (bb_arr[:, 0] >= subimg[0]) & (bb_arr[:, 2] <= subimg[2]) & (bb_arr[:, 1] >= subimg[1]) & (
                bb_arr[:, 3] <= subimg[3])
        a = bb_arr[condition]
        return np.stack([a[:, 0] - subimg[0], a[:, 1] - subimg[1], a[:, 2] - subimg[0], a[:, 3] - subimg[1]], axis=1), \
               lbl_arr[condition]

    def get_subimgs(self):
        """
        Get list of patchs for all slides.

        :return: (imgs, bboxes, labels)
        """

        subimgs = []
        for slide, bboxes_sl in zip(self.slides, self.bboxes):
            bb_arr = np.array([i[0] for i in bboxes_sl])
            lbl_arr = np.array([i[1] for i in bboxes_sl])
            for bbox in tqdm(bboxes_sl):
                subimg = self.get_random_crop_around(slide, bbox[0])
                other_bboxes, other_labels = self.get_all_landed_bboxes(subimg, bb_arr, lbl_arr)
                subimg = slide.get_patch(*subimg[0:2])
                subimg = tf.convert_to_tensor(subimg, dtype=tf.float32)
                other_bboxes = tf.convert_to_tensor(other_bboxes, dtype=tf.float32)
                other_labels = tf.convert_to_tensor(other_labels, dtype=tf.int32)
                subimgs.append({'image': subimg, 'objects': {'bbox': other_bboxes / 255, 'label': other_labels}})
        return subimgs

    def generator(self):
        """
        Finite dataset generator (for validation)

        :return: dataset object
        """
        yield from self.subimgs

    def infinite_generator(self):
        """
        Infinite dataset generator (for traininig)

        :return: dataset object
        """
        while True:
            n_slide = np.random.randint(0, len(self.slides))
            slide = self.slides[n_slide]
            n_bbox = np.random.randint(0, len(self.bboxes[n_slide]))
            bbox = self.bboxes[n_bbox]
            subimg = self.get_random_crop_around(slide, bbox[0])
            other_bboxes, other_labels = self.get_all_landed_bboxes(subimg, self.bb_arr[n_slide], self.lbl_arr[n_slide])
            subimg = slide.get_patch(*subimg[0:2])
            subimg = tf.convert_to_tensor(subimg, dtype=tf.float32)
            other_bboxes = tf.convert_to_tensor(other_bboxes, dtype=tf.float32)
            other_labels = tf.convert_to_tensor(other_labels, dtype=tf.int32)
            yield {'image': subimg, 'objects': {'bbox': other_bboxes / 255, 'label': other_labels}}
