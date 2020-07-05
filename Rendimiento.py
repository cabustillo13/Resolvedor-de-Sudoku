import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from skimage import io, color, img_as_float, filters
from skimage.feature import hog
import cv2
import mahotas

def extraccion(image):
    
    ##PRE PROCESAMIENTO
    image = cv2.resize(image, (500, 400))         #Convertir la imagen de 1920x1080 a 500x4
    aux = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convertir a escala de grises
    
    ##FILTRACION
    aux = cv2.GaussianBlur(aux, (3, 3), 0)   #Aplicar filtro gaussiano
    aux = filters.sobel(aux)                 #Aplicar filtro Sobel o Laplaciano
    
    ##SEGMENTACION
    ##Solo funciona para imagenes cortadas -> porque sino el fondo afecta mucho el objeto dentro de la imagen
    #ret, th = cv2.threshold(aux, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #aux= th
        
    ##EXTRACCION DE RASGOS
    #haralick=mahotas.features.haralick(aux).mean(axis=0)
    hu = cv2.HuMoments(cv2.moments(aux)).flatten()
    
    ##ANALISIS DE LAS CARACTERISTICAS
    "Para 2 elementos"
    #PARA MOMENTOS DE HU
    #return aux, [hu[0], hu[1]]
    ##PARA HARALICK
    #return aux, [haralick[0], haralick[1]]
    
    "Para 3 elementos"
    #PARA MOMENTOS DE HU
    return aux, [hu[0], hu[1], hu[3]]
    #PARA HARALICK
    #return aux, [haralick[0], haralick[1], haralick[3]]
    
    "Para 4 elementos"
    #PARA HARALICK
    #return aux, [haralick[0], haralick[1], haralick[2],haralick[3]]
    #PARA MOMENTOS DE HU
    #return aux, [hu[0], hu[1], hu[2], hu[3]]
    
    "Para todos los elementos -> datos en crudo"
    #PARA MOMENTOS DE HU
    #return aux, hu
    #PARA HARALICK
    #return aux, haralick
    
    "Para 8 elementos"
    #return aux, [haralick[2], haralick[3], haralick[4], haralick[5], haralick[6], haralick[7], haralick[9], haralick[11]]
    
    "Hu + Haralick"
    #return aux, [hu[0], hu[1], hu[3], haralick[0], haralick[1],haralick[3]]

#Elemento de ferreteria
class Elemento:
    def __init__(self):
        self.pieza = None
        self.image = None
        self.caracteristica = []
        self.distancia = 0

#Analisis de la base de datos (YTrain)
##Entrenamiento de la base de datos
def analisis_de_datos():

    tornillo = io.ImageCollection('./Data Base/YTrain/YTornillos/*.png:./Data Base/YTrain/YTornillos/*.jpg')
    tuerca = io.ImageCollection('./Data Base/YTrain/YTuercas/*.png:./Data Base/YTrain/YTuercas/*.jpg')
    arandela = io.ImageCollection('./Data Base/YTrain/YArandelas/*.png:./Data Base/YTrain/YArandelas/*.jpg')
    clavo = io.ImageCollection('./Data Base/YTrain/YClavos/*.png:./Data Base/YTrain/YClavos/*.jpg')
    
    datos = []
    i = 0

    # Analisis de tornillos en la base de datos
    iter = 0
    for objeto in tornillo:
        datos.append(Elemento())
        datos[i].pieza = 'tornillo'
        datos[i].image, datos[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Tornillo OK")

    # Analisis de tuercas en base de datos
    iter = 0
    for objeto in tuerca:
        datos.append(Elemento())
        datos[i].pieza = 'tuerca'
        datos[i].image, datos[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Tuerca OK")

    # Analisis de arandelas en la base de datos
    iter = 0
    for objeto in arandela:
        datos.append(Elemento())
        datos[i].pieza = 'arandela'
        datos[i].image, datos[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Arandela OK")
    
    # Analisis de clavos en la base de datos
    iter = 0
    for objeto in clavo:
        datos.append(Elemento())
        datos[i].pieza = 'clavo'
        datos[i].image, datos[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Clavo OK")

    print("Analisis de todos los objetos en YTrain completo")
    return datos

##Prueba de la base de datos (Yprueba)
def analisis_de_prueba():

    tornillo_prueba = io.ImageCollection('./Data Base/YTest/YTornillos/*.png:./Data Base/YTest/YTornillos/*.jpg')
    tuerca_prueba = io.ImageCollection('./Data Base/YTest/YTuercas/*.png:./Data Base/YTest/YTuercas/*.jpg')
    arandela_prueba = io.ImageCollection('./Data Base/YTest/YArandelas/*.png:./Data Base/YTest/YArandelas/*.jpg')
    clavo_prueba = io.ImageCollection('./Data Base/YTest/YClavos/*.png:./Data Base/YTest/YClavos/*.jpg')
    
    prueba = []
    i = 0

    # Analisis de tornillos en base de datos
    iter = 0
    for objeto in tornillo_prueba:
        prueba.append(Elemento())
        prueba[i].pieza = 'tornillo'
        prueba[i].image, prueba[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Tornillo OK")

    # Analisis de tuercas en base de datos
    iter = 0
    for objeto in tuerca_prueba:
        prueba.append(Elemento())
        prueba[i].pieza = 'tuerca'
        prueba[i].image, prueba[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Tuerca OK")

    # Analisis de arandelas en la base de datos
    iter = 0
    for objeto in arandela_prueba:
        prueba.append(Elemento())
        prueba[i].pieza = 'arandela'
        prueba[i].image, prueba[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Arandela OK")
    
    # Analisis de clavos en la base de datos
    iter = 0
    for objeto in clavo_prueba:
        prueba.append(Elemento())
        prueba[i].pieza = 'clavo'
        prueba[i].image, prueba[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Clavo OK")

    print("Testeo de todos los objetos en YTest completo")
    return prueba

#KNN
def knn(k, datos, prueba):

    correct = 0

    for t in prueba:

        for element in datos:
            sum = 0
            i = 0
            for ft in (element.caracteristica):
                sum = sum + np.power(np.abs((t.caracteristica[i]) - ft), 2)
                i += 1

            element.distancia = np.sqrt(sum)

        # Ordenamiento de burbuja
        swap = True
        while (swap):
            swap = False
            for i in range(1, len(datos)-1) :
                if (datos[i-1].distancia > datos[i].distancia):
                    aux = datos[i]
                    datos[i] = datos[i-1]
                    datos[i-1] = aux
                    swap = True

        eval = [0, 0, 0, 0]

        for i in range(0, k):

            if (datos[i].pieza == 'tornillo'):
                eval[0] += 10

            if (datos[i].pieza == 'tuerca'):
                eval[1] += 10

            if (datos[i].pieza == 'arandela'):
                eval[2] += 10
                
            if (datos[i].pieza == 'clavo'):
                eval[3] += 10
                

        aux = eval[0]
        if (aux < eval[1]):
            aux = eval[1]
        if (aux < eval[2]):
            aux = eval[2]
        if (aux < eval[3]):
            aux = eval[3]

        if (aux == eval[0]):
            pieza = 'tornillo'
        if (aux == eval[1]):
            pieza = 'tuerca'
        if (aux == eval[2]):
            pieza = 'arandela'
        if (aux == eval[3]):
            pieza = 'clavo'

        if (t.pieza == pieza):
            correct += 1
         
    return correct

##Rendimiento KNN - Maldicion de dimensionalidad

print("Inicializacion KKN\n")
datos = analisis_de_datos()
prueba = analisis_de_prueba()

MAX = 120
ans = []

for k in range(1, MAX):
    ans.append(knn(k, datos, prueba))

for i in range(0, len(ans)-1):
    ans[i] = ans[i] * 100 / len(prueba)

fig, ax = plt.subplots()
ax.plot(ans)
ax.grid(True)
ax.set_title('Rendimiento vrs cantidad de vecinos k')
plt.ylabel('Predicciones correctas (%)')
plt.xlabel('K')
plt.show()

#K MEANS
#Entrenamiento de KMeans (YTrain)
import random

def entrenamiento_kmeans(datos):

    tornillo_datos = []
    tuerca_datos = []
    arandela_datos = []
    clavo_datos = []

    # MEANS INICIALES
    for element in datos:
        if (element.pieza == 'tornillo'):
            tornillo_datos.append(element)
        if (element.pieza == 'tuerca'):
            tuerca_datos.append(element)
        if (element.pieza == 'arandela'):
            arandela_datos.append(element)
        if (element.pieza == 'clavo'):
            clavo_datos.append(element)

    tornillo_mean = list(random.choice(tornillo_datos).caracteristica)
    tuerca_mean = list(random.choice(tuerca_datos).caracteristica)
    arandela_mean = list(random.choice(arandela_datos).caracteristica)
    clavo_mean = list(random.choice(clavo_datos).caracteristica)

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

        tornillo_datos = []
        tuerca_datos = []
        arandela_datos = []
        clavo_datos = []

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
                tornillo_datos.append(element.caracteristica)
            elif (aux == dist_tuerca):
                tuerca_datos.append(element.caracteristica)
            elif(aux == dist_arandela):
                arandela_datos.append(element.caracteristica)
            elif(aux == dist_clavo):
                clavo_datos.append(element.caracteristica)

        # ACTUALIZACION
        sum_tornillo = [0, 0, 0]
        for obj1 in tornillo_datos:
            sum_tornillo[0] += obj1[0]
            sum_tornillo[1] += obj1[1]
            sum_tornillo[2] += obj1[2]

        sum_tuerca = [0, 0, 0]
        for obj2 in tuerca_datos:
            sum_tuerca[0] += obj2[0]
            sum_tuerca[1] += obj2[1]
            sum_tuerca[2] += obj2[2]

        sum_arandela = [0, 0, 0]
        for obj3 in arandela_datos:
            sum_arandela[0] += obj3[0]
            sum_arandela[1] += obj3[1]
            sum_arandela[2] += obj3[2]
            
        sum_clavo = [0, 0, 0]
        for obj4 in clavo_datos:
            sum_clavo[0] += obj4[0]
            sum_clavo[1] += obj4[1]
            sum_clavo[2] += obj4[2]

        tornillo_mean[0] = sum_tornillo[0] / len(tornillo_datos)
        tornillo_mean[1] = sum_tornillo[1] / len(tornillo_datos)
        tornillo_mean[2] = sum_tornillo[2] / len(tornillo_datos)

        tuerca_mean[0] = sum_tuerca[0] / len(tuerca_datos)
        tuerca_mean[1] = sum_tuerca[1] / len(tuerca_datos)
        tuerca_mean[2] = sum_tuerca[2] / len(tuerca_datos)

        arandela_mean[0] = sum_arandela[0] / len(arandela_datos)
        arandela_mean[1] = sum_arandela[1] / len(arandela_datos)
        arandela_mean[2] = sum_arandela[2] / len(arandela_datos) 
        
        clavo_mean[0] = sum_clavo[0] / len(clavo_datos)
        clavo_mean[1] = sum_clavo[1] / len(clavo_datos)
        clavo_mean[2] = sum_clavo[2] / len(clavo_datos) 
        
        # CONDICION DE SALIDA
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
        
    return [tornillo_mean, tuerca_mean, arandela_mean, clavo_mean]

#Testeo de KMeans (YTest)
def kmeans(prueba, means):
    
    tornillo_mean = means[0]
    tuerca_mean = means[1]
    arandela_mean = means[2]
    clavo_mean = means[3]
    
    correct = 0

    for t in prueba:

        sum_tornillo = 0
        sum_tuerca = 0
        sum_arandela = 0
        sum_clavo = 0

        for i in range(0, len(t.caracteristica)-1):
            sum_tornillo += np.power(np.abs(t.caracteristica[i] - tornillo_mean[i]), 2)
            sum_tuerca += np.power(np.abs(t.caracteristica[i] - tuerca_mean[i]), 2)
            sum_arandela += np.power(np.abs(t.caracteristica[i] - arandela_mean[i]), 2)
            sum_clavo += np.power(np.abs(t.caracteristica[i] - clavo_mean[i]), 2)

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
            pieza = 'tornillo'
        if (aux == dist_tuerca):
            pieza = 'tuerca'
        if (aux == dist_arandela):
            pieza = 'arandela'
        if (aux == dist_clavo):
            pieza = 'clavo'
        
        if (t.pieza == pieza):
            correct += 1
    
    return correct

##RENDIMIENTO
print("\nInicializacion de KMeans\n")
datos = analisis_de_datos()
prueba = analisis_de_prueba()

means = entrenamiento_kmeans(datos)

#Por mas que varies MAX siempre deberias obtener una linea recta horizontal
MAX = 50

ans = []

for i in range(0, MAX):
    ans.append(kmeans(prueba, means))
    
for i in range(0, len(ans)):
    ans[i] = ans[i] * 100 / len(prueba)

fig, ax = plt.subplots()
ax.plot(ans)
ax.grid(True)
ax.set_title('Rendimiento de KMeans en diferentes ejecuciones')
plt.ylabel('Predicciones correctas (%)')
plt.xlabel('# de ejecucion')
plt.show()

 
 
