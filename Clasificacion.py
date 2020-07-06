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
    #haralick=mahotas.features.haralick(aux).mean(axis=0)
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
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

datos = []
i = 0

# Analisis de la casilla vacia
iter = 0
for objeto in punto:
    datos.append(Elemento())
    datos[i].pieza = 'Punto'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='y', marker='o')
    i += 1
    iter += 1
print("Puntos OK")

# Analisis del numero uno
iter = 0
for objeto in uno:
    datos.append(Elemento())
    datos[i].pieza = 'Uno'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='r', marker='o')
    i += 1
    iter += 1
print("Uno OK")

# Analisis del numero dos
iter = 0
for objeto in dos:
    datos.append(Elemento())
    datos[i].pieza = 'Dos'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='b', marker='o')
    i += 1
    iter += 1
print("Dos OK")

# Analisis del numero tres
iter = 0
for objeto in tres:
    datos.append(Elemento())
    datos[i].pieza = 'Tres'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='g', marker='o')
    i += 1
    iter += 1
print("Tres OK")

# Analisis del numero cuatro
iter = 0
for objeto in cuatro:
    datos.append(Elemento())
    datos[i].pieza = 'Cuatro'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='c', marker='o')
    i += 1
    iter += 1
print("Cuatro OK")

# Analisis del numero cinco
iter = 0
for objeto in cinco:
    datos.append(Elemento())
    datos[i].pieza = 'Cinco'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='m', marker='o')
    i += 1
    iter += 1
print("Cinco OK")

# Analisis del numero seis
iter = 0
for objeto in seis:
    datos.append(Elemento())
    datos[i].pieza = 'Seis'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='k', marker='o')
    i += 1
    iter += 1
print("Seis OK")

# Analisis del numero siete
iter = 0
for objeto in siete:
    datos.append(Elemento())
    datos[i].pieza = 'Siete'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='lime', marker='o')
    i += 1
    iter += 1
print("Siete OK")

# Analisis del numero ocho
iter = 0
for objeto in ocho:
    datos.append(Elemento())
    datos[i].pieza = 'Ocho'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='aqua', marker='o')
    i += 1
    iter += 1
print("Ocho OK")

# Analisis del numero nueve
iter = 0
for objeto in nueve:
    datos.append(Elemento())
    datos[i].pieza = 'Nueve'
    datos[i].image, datos[i].caracteristica = extraccion(objeto)
    ax.scatter(datos[i].caracteristica[0], datos[i].caracteristica[1], datos[i].caracteristica[2], c='purple', marker='o')
    i += 1
    iter += 1
print("Nueve OK")

ax.grid(True)
ax.set_title("Analisis completo de Train")

yellow_patch = mpatches.Patch(color='yellow', label='Punto')
red_patch = mpatches.Patch(color='red', label='Uno')
blue_patch = mpatches.Patch(color='blue', label='Dos')
green_patch = mpatches.Patch(color='green', label='Tres')
cyan_patch = mpatches.Patch(color='cyan', label='Cuatro')
magenta_patch = mpatches.Patch(color='magenta', label='Cinco')
black_patch = mpatches.Patch(color='black', label='Seis')
lime_patch = mpatches.Patch(color='lime', label='Siete')
aqua_patch = mpatches.Patch(color='aqua', label='Ocho')
purple_patch = mpatches.Patch(color='purple', label='Nueve')

plt.legend(handles=[yellow_patch, red_patch, blue_patch, green_patch, cyan_patch, magenta_patch,black_patch,lime_patch,aqua_patch,purple_patch])

ax.set_xlabel('componente 1')
ax.set_ylabel('componente 2')
ax.set_zlabel('componente 4')

plt.show()

print("Analisis completo de la base de datos de YTrain")
print("Cantidad de imagenes analizadas: ")
print(len(datos))

##############################MODIFICAR ABAJO
# Elemento a evaluar
#Recordar aplicar Transformacion.py cuando se quiera evaluar una nueva imagen.
test = Elemento()
numero = input("Introduce numero de la foto: ")

nombre = './Screenshots/photo'+str(numero)+'.png'
image = io.imread(nombre)

test.image, test.caracteristica = extraccion(image)
test.pieza = 'Arandela' # label inicial 

ax.scatter(test.caracteristica[0], test.caracteristica[1], test.caracteristica[2], c='k', marker='o')
fig
##############################MODIFICAR ARRIBA

#KNN
print("\nInicializacion KNN")
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

print("Prediccion para KNN con K=1: ")    
print(test.pieza)

# Algoritmo de ordenamiento de burbuja-> lo elegi porque es bastante estable
swap = True
while (swap):
    swap = False
    for i in range(1, len(datos)-1) :
        if (datos[i-1].distancia > datos[i].distancia):
            aux = datos[i]
            datos[i] = datos[i-1]
            datos[i-1] = aux
            swap = True
print("\nPredicciones para KNN con K=9: ")            
k = 9
for i in range(k):
    print(datos[i].pieza)

#K MEANS
import random
print("\nInicializacion KMeans")

tornillo_data = []
tuerca_data = []
arandela_data = []
clavo_data = []

for element in datos:
    if (element.pieza == 'Tornillo'):
        tornillo_data.append(element)
    if (element.pieza == 'Tuerca'):
        tuerca_data.append(element)
    if (element.pieza == 'Arandela'):
        arandela_data.append(element)
    if (element.pieza == 'Clavo'):
        clavo_data.append(element)

tornillo_mean = list(random.choice(tornillo_data).caracteristica)
tuerca_mean = list(random.choice(tuerca_data).caracteristica)
arandela_mean = list(random.choice(arandela_data).caracteristica)
clavo_mean = list(random.choice(clavo_data).caracteristica)


fig_means = plt.figure()
ax = fig_means.add_subplot(111, projection='3d')

# fig_means, ax = plt.subplots()
ax.scatter(tornillo_mean[0], tornillo_mean[1], tornillo_mean[2], c='y', marker='o')
ax.scatter(tuerca_mean[0], tuerca_mean[1], tuerca_mean[2], c='r', marker='o')
ax.scatter(arandela_mean[0], arandela_mean[1], arandela_mean[2], c='b', marker='o')
ax.scatter(clavo_mean[0], clavo_mean[1], clavo_mean[2], c='g', marker='o')

ax.grid(True)
ax.set_title("Means")

yellow_patch = mpatches.Patch(color='yellow', label='Tornillo')
red_patch = mpatches.Patch(color='red', label='Tuerca')
blue_patch = mpatches.Patch(color='blue', label='Arandela')
green_patch = mpatches.Patch(color='green', label='Clavo')
plt.legend(handles=[yellow_patch, red_patch, blue_patch, green_patch])

ax.set_xlabel('componente 1')
ax.set_ylabel('componente 2')
ax.set_zlabel('componente 4')

plt.show()

# Asignacion, Actualizacion y Convergencia
tornillo_flag = True
tuerca_flag = True
arandela_flag = True
clavo_flag = True

tornillo_len = [0, 0, 0]
tuerca_len = [0, 0, 0]
arandela_len = [0, 0, 0]
clavo_len = [0, 0, 0]

iter = 0
while (iter < 20):

    tornillo_data = []
    tuerca_data = []
    arandela_data = []
    clavo_data = []

    # ASIGNACION
    for element in datos:
        sum_tornillo = 0
        sum_tuerca = 0
        sum_arandela = 0
        sum_clavo = 0

        for i in range(0, len(element.caracteristica)-1):
            sum_tornillo += np.power(np.abs(tornillo_mean[i] - element.caracteristica[i]), 2)
            sum_tuerca += np.power(np.abs(tuerca_mean[i] - element.caracteristica[i]), 2)
            sum_arandela += np.power(np.abs(arandela_mean[i] - element.caracteristica[i]), 2)
            sum_clavo += np.power(np.abs(clavo_mean[i] - element.caracteristica[i]), 2)

        dist_tornillo = np.sqrt(sum_tornillo)
        dist_tuerca = np.sqrt(sum_tuerca)
        dist_arandela = np.sqrt(sum_arandela)
        dist_clavo = np.sqrt(sum_clavo)
        
        aux = dist_tornillo
        if (dist_tuerca < aux):
            aux = dist_tuerca
        if (dist_arandela < aux):
            aux = dist_arandela
        if (dist_clavo < aux):
            aux = dist_clavo
            
        if (aux == dist_tornillo):
            tornillo_data.append(element.caracteristica)
        elif (aux == dist_tuerca):
            tuerca_data.append(element.caracteristica)
        elif(aux == dist_arandela):
            arandela_data.append(element.caracteristica)
        elif(aux == dist_clavo):
            clavo_data.append(element.caracteristica)
            
    # ACTUALIZACION
    sum_tornillo = [0, 0, 0]
    for b in tornillo_data:
        sum_tornillo[0] += b[0]
        sum_tornillo[1] += b[1]
        sum_tornillo[2] += b[2]

    sum_tuerca = [0, 0, 0]
    for o in tuerca_data:
        sum_tuerca[0] += o[0]
        sum_tuerca[1] += o[1]
        sum_tuerca[2] += o[2]

    sum_arandela = [0, 0, 0]
    for l in arandela_data:
        sum_arandela[0] += l[0]
        sum_arandela[1] += l[1]
        sum_arandela[2] += l[2]

    sum_clavo = [0, 0, 0]
    for p in clavo_data:
        sum_clavo[0] += p[0]
        sum_clavo[1] += p[1]
        sum_clavo[2] += p[2]
        
    tornillo_mean[0] = sum_tornillo[0] / len(tornillo_data)
    tornillo_mean[1] = sum_tornillo[1] / len(tornillo_data)
    tornillo_mean[2] = sum_tornillo[2] / len(tornillo_data)

    tuerca_mean[0] = sum_tuerca[0] / len(tuerca_data)
    tuerca_mean[1] = sum_tuerca[1] / len(tuerca_data)
    tuerca_mean[2] = sum_tuerca[2] / len(tuerca_data)

    arandela_mean[0] = sum_arandela[0] / len(arandela_data)
    arandela_mean[1] = sum_arandela[1] / len(arandela_data)
    arandela_mean[2] = sum_arandela[1] / len(arandela_data)
    
    clavo_mean[0] = sum_clavo[0] / len(clavo_data)
    clavo_mean[1] = sum_clavo[1] / len(clavo_data)
    clavo_mean[2] = sum_clavo[1] / len(clavo_data)
    
    #print("Tornillo  Tuerca  Arandela  Clavo")
    #print(len(tornillo_data), len(tuerca_data), len(arandela_data), len(clavo_data))
    
    # CONVERGENCIA Y CONDICION DE SALIDA
    
    if (tornillo_mean == tornillo_len):
        tornillo_flag = False
    else:
        tornillo_len = tornillo_mean

    if (tuerca_mean == tuerca_len):
        tuerca_flag = False
    else:
        tuerca_len = tuerca_mean

    if (arandela_mean == arandela_len):
        arandela_flag = False
    else:
        arandela_len = arandela_mean
            
    if (clavo_mean == clavo_len):
        clavo_flag = False
    else:
        clavo_len = clavo_mean

    iter += 1
    
# Ubicacion de los means finales
ax.scatter(tornillo_mean[0], tornillo_mean[1], tornillo_mean[2], c='k', marker='o')
ax.scatter(tuerca_mean[0], tuerca_mean[1], tuerca_mean[2], c='k', marker='o')
ax.scatter(arandela_mean[0], arandela_mean[1], arandela_mean[2], c='k', marker='o')
ax.scatter(clavo_mean[0], clavo_mean[1], clavo_mean[2], c='k', marker='o')

print("Ubicacion de los means finales")
print("Tornillo  Tuerca  Arandela  Clavo")
print(len(tornillo_data), len(tuerca_data), len(arandela_data), len(clavo_data))
fig_means

##Mean mas cercano
sum_tornillo = 0
sum_tuerca = 0
sum_arandela = 0
sum_clavo = 0

for i in range(0, len(test.caracteristica)-1):
    sum_tornillo += np.power(np.abs(test.caracteristica[i] - tornillo_mean[i]), 2)
    sum_tuerca += np.power(np.abs(test.caracteristica[i] - tuerca_mean[i]), 2)
    sum_arandela += np.power(np.abs(test.caracteristica[i] - arandela_mean[i]), 2)
    sum_clavo += np.power(np.abs(test.caracteristica[i] - clavo_mean[i]), 2)

dist_tornillo = np.sqrt(sum_tornillo)
dist_tuerca = np.sqrt(sum_tuerca)
dist_arandela = np.sqrt(sum_arandela)
dist_clavo = np.sqrt(sum_clavo)

print("\nMean mas cercano")
print("Tornillo  Tuerca  Arandela  Clavo")
print(dist_tornillo, dist_tuerca, dist_arandela, dist_clavo)

aux = dist_tornillo
if (dist_tuerca < aux):
    aux = dist_tuerca
if (dist_arandela < aux):
    aux = dist_arandela
if (dist_clavo < aux):
    aux = dist_clavo

if (aux == dist_tornillo):
    test.pieza = 'Tornillo'
elif (aux == dist_tuerca):
    test.pieza = 'Tuerca'
elif(aux == dist_arandela):
    test.pieza = 'Arandela'
elif(aux == dist_clavo):
    test.pieza = 'Clavo'

print("\nPrediccion para KMeans: ")
print(test.pieza)
