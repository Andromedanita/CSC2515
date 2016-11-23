import numpy            as np
import matplotlib.pylab as plt
from   PIL              import Image
from   sknn.mlp         import Classifier, Convolution, Layer
import csv
import os
import sys


num = int(sys.argv[1])

def load_image():
    folder = "/Users/anita/Documents/Grad_Second_Year/CSC2515/assignment3/411a3/train/"
    filename = os.listdir(folder)[:num]
    num_figs = len(filename)
    all_pixels = np.zeros((num_figs,128,128,3))  #np.zeros((num_figs, 16384,3))
    for i in range(num_figs):
        im  = Image.open(folder+filename[i])#.convert('L')
        pixel_values  = np.array(im.getdata())
        pixel_values  = np.reshape(pixel_values,(128,128,3))
        all_pixels[i] = pixel_values
    return all_pixels


nn = Classifier(
    layers=[Convolution("Rectifier", channels=8, kernel_shape=(3,3)),Layer('Softmax')],
    learning_rate=0.01,n_iter=50)

all_pixels = load_image()

with open('/Users/anita/Documents/Grad_Second_Year/CSC2515/assignment3/411a3/train.csv','rb') as f:
    reader     = csv.reader(f)
    your_list  = list(reader)

your_list = np.array(your_list)
ylabel    = your_list[1:,1].astype(int)[:num]

# balancing weights on classes
#w_train = np.array((all_pixels.shape[0],))
w_train = np.zeros(num)

w_train[ylabel == 1] = 0.414
w_train[ylabel == 2] = 0.448
w_train[ylabel == 3] = 1.25
w_train[ylabel == 4] = 2.155
w_train[ylabel == 5] = 0.523
w_train[ylabel == 6] = 9.61
w_train[ylabel == 7] = 46.05
w_train[ylabel == 8] = 18.23 

nn.fit(all_pixels, ylabel, w_train)

#nn.fit(all_pixels, ylabel)
nn.score(all_pixels, ylabel)
