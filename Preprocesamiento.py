##PARA CORTAR UNA SOLA IMAGEN
import cv2
import numpy as np

#Path absoluto
path="/home/carlos/Documentos/Otros/Resolvedor-de-Sudoku/Screenshots/"
#Imagen a analizar
imagen = cv2.imread(path+"photo1.png")

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
    
    new_path = "/home/carlos/Documentos/Otros/Resolvedor-de-Sudoku/Imagenes/"
    cv2.imwrite(new_path + fileb, imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#######################################################################
################                    El main                 ###########
#######################################################################

if __name__ == '__main__':    
    
    ##PREPROCESAMIENTO -> Aca recordamos los bordes, publicidad y todo las imagenes ajenas al tablero del sudoku
    imagen = recortar(imagen,218)
    imagen = rotarImagen(imagen,180)
    imagen = recortar(imagen,348)
    imagen = rotarImagen(imagen,180)
    
    ##RECORTAR CADA CUADRADITO DE FORMA INDIVIDUAL
    contador=0
    w=0
    for j in range(9):
        
        h=0
        for i in range(9):
            #A prueba y error determine los valores de los lados de los cuadrados
            nombre = "prueba"+ str(contador) + ".png"
            imagen1 = recortarCuadrado(imagen,0,h,75,80)
            
            #Procedemos a guardarla
            guardar(imagen1,nombre)
            h=h+80
            contador=contador+1
        w=w+75
        
    
