import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
#Grayscale histogram of the original image
basename='image/s7x223y835.jpg'
file_name = os.path.splitext(basename)[0]
img = cv.imread(basename)
grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
max0=np.max(grayImage)
min0=np.min(grayImage)
hist0,bins=np.histogram(grayImage,bins=(max0-min0))    
if min0>0:
    for i in range(1,min0+1):
        hist0=np.insert(hist0,0,[0])
if len(hist0)<255:
    for i in range(len(hist0),256):
        hist0=np.append(hist0,[0])
hist0[254]=0
for i in range(1,len(hist0)+1):                
    with open('histb.txt', 'a') as file:
        file.write(str(i-1)+','+str(int(hist0[i-1]))+'\n') 
plt.plot(hist0)      
plt.xticks(range(0,240,20))
plt.savefig(file_name+'histb.png')
plt.show()
cv.waitKey(0)