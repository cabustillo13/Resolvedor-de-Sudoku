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
 
#######################################################################
################    Recortar franja de arriba imagen        ###########
#######################################################################
def recortar(imagen,x):
    #x determina hasta donde cortar
    #fileb determina con que nombre lo vamos a guardar a la imagen
    #Determinar dimensiones de la imagen
    height, width, channels = imagen.shape

    crop_img = imagen[x:height, 0:width]
    return crop_img

#######################################################################
################    Recortar casilla/cuadrado               ###########
#######################################################################
def recortarCuadrado(imagen,x,y,h,w):
    
    #Cada lado del cuadrado/casilla tiene un lado de longitud 10
    crop_img = imagen[x:(x+h), y:(y+w)]
    return crop_img

#######################################################################
################                Guardar imagen              ###########
#######################################################################
def guardar(imagen,fileb):
    
    new_path = "./Imagenes/"
    cv2.imwrite(new_path + fileb, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#######################################################################
################    Recortar bordes de la imagen            ###########
#######################################################################
def recortarBorde(imagen):
    #Determinar dimensiones de la imagen
    height, width, channels = imagen.shape

    crop_img = imagen[12:height-12, 12:width-12]
    return crop_img
