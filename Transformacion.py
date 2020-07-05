##PARA CORTAR UNA SOLA IMAGEN
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
def recortar(imagen):
    #Determinar dimensiones de la imagen
    height, width, channels = imagen.shape

    crop_img = imagen[10:height-10, 10:width-10]
    return crop_img
 
#######################################################################
################                Guardar imagen              ###########
#######################################################################
def guardar(imagen,fileb):
    
    new_path = "/home/carlos/Documentos/Otros/Resolvedor-de-Sudoku/ImagenesRecortadas/"
    cv2.imwrite(new_path + fileb, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Path absoluto
path="/home/carlos/Documentos/Otros/Resolvedor-de-Sudoku/Imagenes/"

if __name__ == '__main__':
    
    for x in range(81):
        #Imagen a analizar
        nombre = "prueba" + str(x) + ".png"
        imagen = cv2.imread(path+nombre)
        imagen = recortar(imagen)
        guardar(imagen,nombre)
