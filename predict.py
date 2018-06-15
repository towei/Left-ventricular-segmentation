import numpy as np
from keras.models import model_from_json
import os
import scipy.misc


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
        label_dir = os.path.join(data_dir, "y1")
        image_dir = os.path.join(data_dir, "x1")
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
print('result.shape:')
print(img.shape)

# calculation Dice parameters
ind = 0
for i in range(0,161):
    for j in range (0,256):
        for z in range (0,256):
            if img[i,j,z,0] != 0:
                if x_test[i,j,z,0] != 0:
                    ind = ind+1


cTPTN = (img == y_test)
cFPFN = (img != y_test)

cTPTN = cTPTN.astype(np.int32)
cFPFN = cFPFN.astype(np.int32)

cTPTN = cTPTN[...,0]
cFPFN = cFPFN[...,0]

TPTN = np.cumsum(cTPTN)
TPTN = TPTN[-1]
FPFN = np.cumsum(cFPFN)
FPFN = FPFN[-1]

TP = ind
Dice = 2.0*TP/(2.0*TP+FPFN)

print('Dice:',Dice)
print('Accuracy:',1-1.0*FPFN/(TPTN+FPFN))

