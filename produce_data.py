import random
import os
import skimage.io as io


# enter path and enter images
Ndir = os.getcwd()
pimg_dir = os.path.join(Ndir, 'x')
name = os.listdir(pimg_dir)

# random disorder of the original data
random.shuffle(name)
print(len(name))
print(name)

# divided into 5 groups
glen = int(len(name)/5)
g1 = list()
g2 = list()
g3 = list()
g4 = list()
g5 = list()

for i in range(len(name)):
    if i < glen:
        g1.append(name[i])
    elif i < glen*2:
        g2.append(name[i])
    elif i < glen*3:
        g3.append(name[i])
    elif i < glen*4:
        g4.append(name[i])
    elif i < glen*5:
        g5.append(name[i])

print(g1)
print(g2)
print(g3)
print(g4)
print(g5)

# image and label correspondence
for i in range(len(name)):
    if i < glen:
        x_name = '/home/xdml/PycharmProjects/dltower/CSG/x/' + g1[i]
        xd_name = '/home/xdml/PycharmProjects/dltower/CSG/x1/' + g1[i]
        y_name = '/home/xdml/PycharmProjects/dltower/CSG/y/' + g1[i]
        yd_name = '/home/xdml/PycharmProjects/dltower/CSG/y1/' + g1[i]
    elif i < glen*2:
        x_name = '/home/xdml/PycharmProjects/dltower/CSG/x/' + g2[i-glen]
        xd_name = '/home/xdml/PycharmProjects/dltower/CSG/x2/' + g2[i-glen]
        y_name = '/home/xdml/PycharmProjects/dltower/CSG/y/' + g2[i-glen]
        yd_name = '/home/xdml/PycharmProjects/dltower/CSG/y2/' + g2[i-glen]
    elif i < glen*3:
        x_name = '/home/xdml/PycharmProjects/dltower/CSG/x/' + g3[i-glen*2]
        xd_name = '/home/xdml/PycharmProjects/dltower/CSG/x3/' + g3[i-glen*2]
        y_name = '/home/xdml/PycharmProjects/dltower/CSG/y/' + g3[i-glen*2]
        yd_name = '/home/xdml/PycharmProjects/dltower/CSG/y3/' + g3[i-glen*2]
    elif i < glen*4:
        x_name = '/home/xdml/PycharmProjects/dltower/CSG/x/' + g4[i-glen*3]
        xd_name = '/home/xdml/PycharmProjects/dltower/CSG/x4/' + g4[i-glen*3]
        y_name = '/home/xdml/PycharmProjects/dltower/CSG/y/' + g4[i-glen*3]
        yd_name = '/home/xdml/PycharmProjects/dltower/CSG/y4/' + g4[i-glen*3]
    elif i < glen*5:
        x_name = '/home/xdml/PycharmProjects/dltower/CSG/x/' + g5[i-glen*4]
        xd_name = '/home/xdml/PycharmProjects/dltower/CSG/x5/' + g5[i-glen*4]
        y_name = '/home/xdml/PycharmProjects/dltower/CSG/y/' + g5[i-glen*4]
        yd_name = '/home/xdml/PycharmProjects/dltower/CSG/y5/' + g5[i-glen*4]
    # turn the original image into a grayscale image
    img1 = io.imread(x_name,as_grey=True)
    io.imsave(xd_name,img1)
    img2 = io.imread(y_name,as_grey=True)
    io.imsave(yd_name,img2)

print("finshed")