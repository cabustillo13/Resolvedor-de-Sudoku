import cv2
import numpy as np
import Functions

#Relative path 
path="./Images/"

if __name__ == '__main__':
    
    for x in range(81):
        #Image to analize
        nameImage = "image" + str(x) + ".png"
        image = cv2.imread(path+nameImage)
        image = Functions.cropBorder(image)
        Functions.saveImage(image,nameImage)
