import cv2
import numpy as np

#######################################################################
################    Funcion para rotar imagen               ###########
#######################################################################
def rotarImagen(image, angle):
     image_center = tuple(np.array(image.shape[1::-1]) / 2)
     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
     result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
     return result
 
