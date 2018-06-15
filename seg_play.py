import numpy as np
from keras.models import model_from_json
import os
import scipy.misc
from skimage import io


# Super parameter setting
height = 256
width = 256
Ndir = os.getcwd()

# Load trained network model and parameters
model = model_from_json(open('my_model_architecture3.json').read())
model.load_weights('my_model_weights3.h5')


# read database class
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

                    # image preprocessing
                    image = image[..., None]/255

                    label = label[..., None]
                    label = label>1
                    label = label*255
                    label = label.astype(np.int32)

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

# enter path and enter image
tedata_dir = os.path.join(Ndir, 'Data/test')
test_data = GetData(tedata_dir)

x_test = test_data.images
y_test = test_data.labels


# Segmentation of test data with a trained network
result = model.predict(x_test, verbose=0)

# The processing of the segmentation results is easier to compare
img = result
img = img>1
img = img*255
img = img.astype(np.int32)
img = img[...,0]

# save testing images
pl1 = img[0,:,:]
io.imsave('/home/xdml/PycharmProjects/dltower/CSG/Data/test/play/1.jpg', pl1)
pl2 = img[1,:,:]
io.imsave('/home/xdml/PycharmProjects/dltower/CSG/Data/test/play/2.jpg', pl2)
pl3 = img[2,:,:]
io.imsave('/home/xdml/PycharmProjects/dltower/CSG/Data/test/play/3.jpg', pl3)
pl4 = img[3,:,:]
io.imsave('/home/xdml/PycharmProjects/dltower/CSG/Data/test/play/4.jpg', pl4)

print("finish")
