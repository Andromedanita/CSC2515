import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from sknn.mlp import Classifier, Convolution, Layer
import csv
import os
import sys

plt.ion()

num = int(sys.argv[1])

def load_image():
    folder = "/Users/anita/Documents/Grad_Second_Year/CSC2515/assignment3/411a3/train/"
    filename = os.listdir(folder)[:num]
    num_figs = len(filename)
    all_pixels = np.zeros((num_figs, 128,128,3))  #np.zeros((num_figs, 16384,3))
    for i in range(num_figs):
        print i
        im  = Image.open(folder+filename[i])#.convert('L')
        #pix = im.load()
        pixel_values = np.array(im.getdata())
        pixel_values = np.reshape(pixel_values,(128,128,3))
        all_pixels[i] = pixel_values
        plt.figure(1)
        #plt.imshow(pixel_values.reshape(128,128), interpolation='nearest')
    return all_pixels


nn = Classifier(
    layers=[Convolution("Rectifier", channels=8, kernel_shape=(3,3)),Layer("Softmax")],
    learning_rate=0.02,n_iter=50)

all_pixels = load_image()


with open('/Users/anita/Documents/Grad_Second_Year/CSC2515/assignment3/411a3/train.csv','rb') as f:
    reader     = csv.reader(f)
    your_list = list(reader)

your_list = np.array(your_list)
ylabel = your_list[1:,1].astype(int)[:num]
    
#ylabel = np.array([1,2,3,4,5])
nn.fit(all_pixels, ylabel)

