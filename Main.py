#Codigo limpio de Clasificacion.py

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['image.cmap'] = 'gray'
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, color, img_as_float, filters
from skimage.feature import hog
import cv2
import mahotas
    
def extraccion(image):
    
    ##TRANSFORMACION
    #Recordar hacer la transformacion de la imagen con el programa Transformacion.py
    image = cv2.resize(image, (500, 400))         #Convertir la imagen de 1220x1080 a 500x400
    
    ##PRE PROCESAMIENTO
    aux = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convertir a escala de grises
    
    ##FILTRACION
    aux = cv2.GaussianBlur(aux, (3, 3), 0)   #Aplicar filtro gaussiano
    #aux = filters.sobel(aux)                 #Aplicar filtro Sobel o Laplaciano
    
    ##SEGMENTACION
    ret, th = cv2.threshold(aux, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    aux= th
        
    ##EXTRACCION DE RASGOS
    hu = cv2.HuMoments(cv2.moments(aux)).flatten()
    
    ##ANALISIS DE LAS CARACTERISTICAS
    #PARA MOMENTOS DE HU
    return aux, [hu[0], hu[1], hu[3]]

#Analisis de la base de datos (YTrain)
##Entrenamiento de la base de datos 
punto = io.ImageCollection('./Imagenes/Train/YPunto/*.png:./Imagenes/Train/YPunto/*.jpg')
uno = io.ImageCollection('./Imagenes/Train/Y1/*.png:./Imagenes/Train/Y1/*.jpg')
dos = io.ImageCollection('./Imagenes/Train/Y2/*.png:./Imagenes/Train/Y2/*.jpg')
tres = io.ImageCollection('./Imagenes/Train/Y3/*.png:./Imagenes/Train/Y3/*.jpg')
cuatro = io.ImageCollection('./Imagenes/Train/Y4/*.png:./Imagenes/Train/Y4/*.jpg')
cinco = io.ImageCollection('./Imagenes/Train/Y5/*.png:./Imagenes/Train/Y5/*.jpg')
seis = io.ImageCollection('./Imagenes/Train/Y6/*.png:./Imagenes/Train/Y6/*.jpg')
siete = io.ImageCollection('./Imagenes/Train/Y7/*.png:./Imagenes/Train/Y7/*.jpg')
ocho = io.ImageCollection('./Imagenes/Train/Y8/*.png:./Imagenes/Train/Y8/*.jpg')
nueve = io.ImageCollection('./Imagenes/Train/Y9/*.png:./Imagenes/Train/Y9/*.jpg')
    
#Elemento de sudoku
class Elemento:
    def __init__(self):
        self.pieza = None
        self.image = None
        self.caracteristica = []
        self.distancia = 0
        
#Analisis de datos
datos = []
i = 0

# Analisis de la casilla vacia
iter = 0
for objeto in punto:
    datos.append(Elemento())
    datos[i].pieza = '.'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("Puntos OK")

# Analisis del numero uno
iter = 0
for objeto in uno:
    datos.append(Elemento())
    datos[i].pieza = '1'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("Uno OK")

# Analisis del numero dos
iter = 0
for objeto in dos:
    datos.append(Elemento())
    datos[i].pieza = '2'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("Dos OK")

# Analisis del numero tres
iter = 0
for objeto in tres:
    datos.append(Elemento())
    datos[i].pieza = '3'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("Tres OK")

# Analisis del numero cuatro
iter = 0
for objeto in cuatro:
    datos.append(Elemento())
    datos[i].pieza = '4'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("Cuatro OK")

# Analisis del numero cinco
iter = 0
for objeto in cinco:
    datos.append(Elemento())
    datos[i].pieza = '5'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("Cinco OK")

# Analisis del numero seis
iter = 0
for objeto in seis:
    datos.append(Elemento())
    datos[i].pieza = '6'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("Seis OK")

# Analisis del numero siete
iter = 0
for objeto in siete:
    datos.append(Elemento())
    datos[i].pieza = '7'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("Siete OK")

# Analisis del numero ocho
iter = 0
for objeto in ocho:
    datos.append(Elemento())
    datos[i].pieza = '8'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("Ocho OK")

# Analisis del numero nueve
iter = 0
for objeto in nueve:
    datos.append(Elemento())
    datos[i].pieza = '9'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    i += 1
    iter += 1
print("Nueve OK")

print("Analisis completo de la base de datos de Train")
#print("Cantidad de imagenes analizadas: ")
#print(len(datos))

#KNN
print("\nInicializacion KNN")

# Elemento a evaluar
#Recordar aplicar Transformacion.py cuando se quiera evaluar una nueva imagen.
test = Elemento()

for numero in range(81):

    nombre = './Imagenes/prueba'+str(numero)+'.png'
    image = io.imread(nombre)

    test.image, test.caracteristica = extraccion(image)
    test.pieza = '1' # label inicial 

    i = 0
    sum = 0
    for ft in datos[0].caracteristica:
        sum = sum + np.power(np.abs(test.caracteristica[i] - ft), 2)
        i += 1
    d = np.sqrt(sum)

    for element in datos:
        sum = 0
        i = 0
        for ft in (element.caracteristica):
            sum = sum + np.power(np.abs((test.caracteristica[i]) - ft), 2)
            i += 1
    
        element.distancia = np.sqrt(sum)
    
        if (sum < d):
            d = sum
            test.pieza = element.pieza

    #print("Prediccion para KNN con K=1: ")    
    #print(test.pieza)
    if (numero == 0): vector =  str(test.pieza)
    else: vector = vector + str(test.pieza) 
    
print(vector)
#Aca trabajamos con manejadores de archivo
archivo = open("vector.txt","w")
archivo.write(vector)
archivo.close()
