from keras.models import Model
from keras.layers import Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers import Dense, Dropout,Input
import numpy as np
import os
import scipy.misc


# Super parameter setting
batch_size = 16
height = 256
width = 256
epochs = 10
Ndir = os.getcwd()

# read database class
class GetData():
    def __init__(self, data_dir):
        images_list = []
        labels_list = []

        self.source_list = []

        examples = 0
        print("loading images")
        label_dir = os.path.join(data_dir, "y")
        image_dir = os.path.join(data_dir, "x")
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
                    label = label * 255
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

# enter path and enter images
trdata_dir = os.path.join(Ndir, 'Data/train')
train_data = GetData(trdata_dir)
test_data = GetData(trdata_dir)

x_train = train_data.images
y_train = train_data.labels
x_test = test_data.images
y_test = test_data.labels


input_shape = x_train.shape[1:]
inputs = Input(shape=input_shape)

print(x_train.shape)


# Build the network
x = Conv2D(64, (3, 3), strides=1,padding='same',input_shape=input_shape,activation='relu')(inputs)
x = Conv2D(64, (3, 3), strides=1,padding='same',activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
x = Conv2D(128, (3, 3), strides=1,padding='same',activation='relu')(x)
x = Conv2D(128, (3, 3), strides=1,padding='same',activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
x = Conv2D(256, (3, 3), strides=1,padding='same',activation='relu')(x)
x = Conv2D(256, (3, 3), strides=1,padding='same',activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
x = Conv2D(512, (7, 7), strides=1,padding='same',activation='relu')(x)
x = Dropout(0.5)(x)
x = Conv2D(512, (1, 1), strides=1,padding='same',activation='relu')(x)
x = Dropout(0.5)(x)
x = Conv2DTranspose(256, (3, 3), strides=1,padding='same',activation='relu')(x)
x = Conv2DTranspose(256, (3, 3), strides=1,padding='same',activation='relu')(x)
x = UpSampling2D(size=(2, 2))(x)
x = Conv2DTranspose(128, (3, 3), strides=1,padding='same',activation='relu')(x)
x = Conv2DTranspose(128, (3, 3), strides=1,padding='same',activation='relu')(x)
x = UpSampling2D(size=(2, 2))(x)
x = Conv2DTranspose(64, (3, 3), strides=1,padding='same',activation='relu')(x)
x = Conv2DTranspose(1, (3, 3), strides=1,padding='same',activation='relu')(x)
x = UpSampling2D(size=(2, 2))(x)
out = Dense(1,activation='sigmoid')(x)


model = Model(inputs=inputs, outputs=out)
# Compile model
model.compile(optimizer='Adam',loss='mse',metrics=None)
# Training model
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1)

# save the trained network structure and parameters
json_string = model.to_json()
open('my_model_architecture10.json','w').write(json_string)
model.save_weights('my_model_weights10.h5')
