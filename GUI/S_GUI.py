import tkinter as tk
import numpy as np
from keras.models import model_from_json
import os
import scipy.misc
from skimage import io

Ndir = os.getcwd()

window = tk.Tk()
window.title('Left ventricular segmentation')
window.geometry('600x750')

canvas = tk.Canvas(window, bg='yellow', height=600, width=600)
image_file1 = tk.PhotoImage(file='img1.png')
image1 = canvas.create_image(22, 20, anchor='nw', image=image_file1)
image_file2 = tk.PhotoImage(file='lab1.png')
image2 = canvas.create_image(578, 20, anchor='ne', image=image_file2)

canvas.pack()

l = tk.Label(window, text='Accuracy', bg='green',font=('Arial', 12), width=20, height=2)
l.pack()
var = tk.StringVar()
l1 = tk.Label(window, textvariable=var,bg='green', font=('Arial', 12), width=20, height=2)
l1.pack()
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
                if not file.endswith(".png"):
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



def play():
    # enter path and enter image
    # tedata_dir = os.path.join(Ndir)
    test_data = GetData(Ndir)

    x_test = test_data.images
    y_test = test_data.labels

    # Segmentation of test data with a trained network
    result = model.predict(x_test, verbose=0)

    # The processing of the segmentation results is easier to compare
    img = result
    img = img > 1
    img = img * 255
    img = img.astype(np.int32)
    img = img[..., 0]

    # save testing images
    pl1 = img[0, :, :]
    io.imsave('/home/xdml/PycharmProjects/dltower/CSG/GUI/pla1.png', pl1)

    y_test1 = y_test[...,0]
    y_test1 = y_test1[0,:,:]

    cTPTN = (img == y_test1)
    cFPFN = (img != y_test1)

    cTPTN = cTPTN.astype(np.int32)
    cFPFN = cFPFN.astype(np.int32)


    TPTN = np.cumsum(cTPTN)
    TPTN = TPTN[-1]
    FPFN = np.cumsum(cFPFN)
    FPFN = FPFN[-1]

    Accuracy = 1-1.0*FPFN/(TPTN+FPFN)

    return(Accuracy)


def seg():
    acc = play()
    filename = tk.PhotoImage(file='pla1.png')
    canvas.image = filename  # <--- keep reference of your image
    canvas.create_image(22, 300, anchor='nw', image=filename)
    var.set(acc)

b = tk.Button(window, text='segment image',width=15, height=2, command=seg).pack()     # 点击按钮式执行的命令

window.mainloop()