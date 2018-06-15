import os
import random

import numpy as np

import scipy.misc

class GetData():
    def __init__(self, data_dir):
        images_list = []
        labels_list = []

        self.source_list = []

        examples = 0
        print("loading images")
        label_dir = os.path.join(data_dir, "Labels")
        image_dir = os.path.join(data_dir, "Images")
        for label_root, dir, files in os.walk(label_dir):
            for file in files:
                if not file.endswith((".png", ".jpg", ".gif")):
                    continue
                try:
                    folder = os.path.relpath(label_root, label_dir)
                    image_root = os.path.join(image_dir, folder)

                    image = scipy.misc.imread(os.path.join(image_root, file))
                    label = scipy.misc.imread(os.path.join(label_root, file))

                    image = image[...,0][...,None]/255

                    label = label[...,0]>1
                    label = label[...,None]
                    label = label.astype(np.int64)

                    images_list.append(image)
                    labels_list.append(label)
                    examples = examples + 1
                except Exception as e:
                    print(e)
        print("finished loading images")
        self.examples = examples
        print("Number of examples found: ", examples)
        self.images = np.array(images_list)
        self.labels = np.array(labels_list)
