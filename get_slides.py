import openslide
from tqdm import tqdm
import os
from data_loader import SlideContainer
from SlideRunner.dataAccess.annotations import *


def get_slides(slidelist_test: list, database: "Database", positive_class: int = 2, negative_class: int = 7,
               basepath: str = 'WSI', size: int = 256):
    """
    Get slides from database.

    :param slidelist_test: ignore list
    :param database: database
    :param positive_class: number of positive classes
    :param negative_class: number of negative classes
    :param basepath: path to database
    :param size: size for patches

    :return: list(tuple(label, bbox)), list(SlideContainer)
    """
    lbl_bbox = list()
    files = list()

    getslides = """SELECT uid, filename FROM Slides"""
    for idx, (currslide, filename) in enumerate(
            tqdm(database.execute(getslides).fetchall(), desc='Loading slides .. ')):
        if (str(currslide) in slidelist_test):  # skip test slides
            continue

        database.loadIntoMemory(currslide)

        slide_path = basepath + os.sep + filename

        slide = openslide.open_slide(str(slide_path))

        level = 0  # slide.level_count - 1
        level_dimension = slide.level_dimensions[level]
        down_factor = slide.level_downsamples[level]

        classes = {positive_class: 1}  # Map non-mitosis to background

        labels, bboxes = [], []
        annotations = dict()
        for id, annotation in database.annotations.items():
            if annotation.deleted or annotation.annotationType != AnnotationType.SPOT:
                continue
            annotation.r = 25
            d = 2 * annotation.r / down_factor
            x_min = (annotation.x1 - annotation.r) / down_factor
            y_min = (annotation.y1 - annotation.r) / down_factor
            x_max = x_min + d
            y_max = y_min + d
            if annotation.agreedClass not in annotations:
                annotations[annotation.agreedClass] = dict()
                annotations[annotation.agreedClass]['bboxes'] = list()
                annotations[annotation.agreedClass]['label'] = list()

            annotations[annotation.agreedClass]['bboxes'].append([int(x_min), int(y_min), int(x_max), int(y_max)])
            annotations[annotation.agreedClass]['label'].append(annotation.agreedClass)

            if annotation.agreedClass in classes:
                label = classes[annotation.agreedClass]

                bboxes.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                labels.append(label)

        if len(bboxes) > 0:
            lbl_bbox.append([(bboxes[i], labels[i]) for i in range(len(bboxes))])
            files.append(SlideContainer(file=slide_path, annotations=annotations, level=level, width=size, height=size,
                                        y=[bboxes, labels]))
    return lbl_bbox, files
