##PARA CORTAR UNA SOLA IMAGEN
import cv2
import numpy as np
import Funciones

#Path relativo
path="./Screenshots/"

#Imagen a analizar
numero = input("Ingrese numero de imagen: ")
imagen = cv2.imread(path+"photo"+str(numero)+".png")

#######################################################################
################                    MAIN                    ###########
#######################################################################

if __name__ == '__main__':    
    
    ##PREPROCESAMIENTO -> Aca recordamos los bordes, publicidad y todo las imagenes ajenas al tablero del sudoku
    imagen = Funciones.recortar(imagen,218)
    imagen = Funciones.rotarImagen(imagen,180)
    imagen = Funciones.recortar(imagen,348)
    imagen = Funciones.rotarImagen(imagen,180)
    
    ##RECORTAR CADA CUADRADITO DE FORMA INDIVIDUAL
    contador=0
    w=0
    for j in range(9):
        
        h=0
        for i in range(9):
            #A prueba y error determine los valores de los lados de los cuadrados
            nombre = "prueba"+ str(contador) + ".png"
            imagen1 = Funciones.recortarCuadrado(imagen,w,h,75,80)
            
            #Procedemos a guardarla
            Funciones.guardar(imagen1,nombre)
            
            h=h+80
            contador=contador+1
        
        #Deduccion que se tomo para lograr el punto exacto donde arranca cada casilla
        w=80*(j+1)
        
