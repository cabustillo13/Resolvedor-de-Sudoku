import cv2
import numpy as np
import Funciones

#Path relativo
path="./Imagenes/"

if __name__ == '__main__':
    
    for x in range(81):
        #Imagen a analizar
        nombre = "prueba" + str(x) + ".png"
        imagen = cv2.imread(path+nombre)
        imagen = Funciones.recortarBorde(imagen)
        Funciones.guardar(imagen,nombre)
