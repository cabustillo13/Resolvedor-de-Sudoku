import cv2
import numpy as np

# Function to rotate the image
def rotateImage(image, angle):
     image_center = tuple(np.array(image.shape[1::-1]) / 2)
     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
     result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
     return result
 
# Function to crop top border in the image
def cropImage(image,x):
    #x determine how far to cut the image
    #fileb determines with what name we are going to save the image
    #Determine image dimensions
    height, width, channels = image.shape

    crop_img = image[x:height, 0:width]
    return crop_img

# Function to crop every box (there are 81 boxes in total) 
def cropBox(image,x,y,h,w):
    
    #Each side of the square / box has a side of length 10
    crop_img = image[x:(x+h), y:(y+w)]
    return crop_img

# Function to save the image 
def saveImage(image,fileb):
    
    new_path = "./Images/"
    cv2.imwrite(new_path + fileb, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to crop all borders of each box
def cropBorder(image):
    #Determine image dimensions
    height, width, channels = image.shape

    crop_img = image[12:height-12, 12:width-12]
    return crop_img
