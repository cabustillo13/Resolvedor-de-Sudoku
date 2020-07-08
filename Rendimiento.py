import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
from skimage import io, color, img_as_float, filters
from skimage.feature import hog
import cv2
import mahotas

def extraccion(image):
    
    ##PRE PROCESAMIENTO
    #image = cv2.resize(image, (60, 55))          #Convertir la imagen a 60x55
    aux = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convertir a escala de grises
    
    ##CONTADOR DE OBJETOS DENTRO DE LA IMAGEN CON ALGORITMO CANNY
    bordes = cv2.Canny(aux, 10, 140)                                                #Estos valores de umbrales se obtuvieron de prueba y error
    ctns, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   #Para OpenCV4
    contornos = len(ctns)
    
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
    #return aux, [hu[0], hu[1], hu[3]]
    return aux, [hu[0], hu[1], contornos]

#Elemento de sudoku
class Elemento:
    def __init__(self):
        self.pieza = None
        self.image = None
        self.caracteristica = []
        self.distancia = 0

#Analisis de la base de datos (Train)
##Entrenamiento de la base de datos
def analisis_de_datos():

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
    
    datos = []
    i = 0

    # Analisis de la casilla vacia en la base de datos
    iter = 0
    for objeto in punto:
        datos.append(Elemento())
        datos[i].pieza = 'punto'
        datos[i].image, datos[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Punto OK")

    # Analisis del numero uno en base de datos
    iter = 0
    for objeto in uno:
        datos.append(Elemento())
        datos[i].pieza = 'uno'
        datos[i].image, datos[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Uno OK")

    # Analisis del numero dos en la base de datos
    iter = 0
    for objeto in dos:
        datos.append(Elemento())
        datos[i].pieza = 'dos'
        datos[i].image, datos[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Dos OK")
    
    # Analisis del numero tres en la base de datos
    iter = 0
    for objeto in tres:
        datos.append(Elemento())
        datos[i].pieza = 'tres'
        datos[i].image, datos[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Tres OK")
    
    # Analisis del numero cuatro en la base de datos
    iter = 0
    for objeto in cuatro:
        datos.append(Elemento())
        datos[i].pieza = 'cuatro'
        datos[i].image, datos[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Cuatro OK")

    # Analisis del numero cinco en base de datos
    iter = 0
    for objeto in cinco:
        datos.append(Elemento())
        datos[i].pieza = 'cinco'
        datos[i].image, datos[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Cinco OK")

    # Analisis del numero seis en la base de datos
    iter = 0
    for objeto in seis:
        datos.append(Elemento())
        datos[i].pieza = 'seis'
        datos[i].image, datos[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Seis OK")
    
    # Analisis del numero siete en la base de datos
    iter = 0
    for objeto in siete:
        datos.append(Elemento())
        datos[i].pieza = 'siete'
        datos[i].image, datos[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Siete OK")

    # Analisis del numero ocho en la base de datos
    iter = 0
    for objeto in ocho:
        datos.append(Elemento())
        datos[i].pieza = 'ocho'
        datos[i].image, datos[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Ocho OK")
    
    # Analisis del numero nueve en la base de datos
    iter = 0
    for objeto in nueve:
        datos.append(Elemento())
        datos[i].pieza = 'nueve'
        datos[i].image, datos[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Nueve OK")

    print("Analisis de todos los objetos de Train completo")
    return datos

##Prueba de la base de datos (Test)
def analisis_de_prueba():

    punto_prueba = io.ImageCollection('./Imagenes/Test/YPunto/*.png:./Imagenes/Test/YPunto/*.jpg')
    uno_prueba = io.ImageCollection('./Imagenes/Test/Y1/*.png:./Imagenes/Test/Y1/*.jpg')
    dos_prueba = io.ImageCollection('./Imagenes/Test/Y2/*.png:./Imagenes/Test/Y2/*.jpg')
    tres_prueba = io.ImageCollection('./Imagenes/Test/Y3/*.png:./Imagenes/Test/Y3/*.jpg')
    cuatro_prueba = io.ImageCollection('./Imagenes/Test/Y4/*.png:./Imagenes/Test/Y4/*.jpg')
    cinco_prueba = io.ImageCollection('./Imagenes/Test/Y5/*.png:./Imagenes/Test/Y5/*.jpg')
    seis_prueba = io.ImageCollection('./Imagenes/Test/Y6/*.png:./Imagenes/Test/Y6/*.jpg')
    siete_prueba = io.ImageCollection('./Imagenes/Test/Y7/*.png:./Imagenes/Test/Y7/*.jpg')
    ocho_prueba = io.ImageCollection('./Imagenes/Test/Y8/*.png:./Imagenes/Test/Y8/*.jpg')
    nueve_prueba = io.ImageCollection('./Imagenes/Test/Y9/*.png:./Imagenes/Test/Y9/*.jpg')
    
    prueba = []
    i = 0

    # Analisis de la casilla vacia en base de datos
    iter = 0
    for objeto in punto_prueba:
        prueba.append(Elemento())
        prueba[i].pieza = 'punto'
        prueba[i].image, prueba[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Punto OK")

    # Analisis del numero uno en base de datos
    iter = 0
    for objeto in uno_prueba:
        prueba.append(Elemento())
        prueba[i].pieza = 'uno'
        prueba[i].image, prueba[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Uno OK")

    # Analisis del numero dos en la base de datos
    iter = 0
    for objeto in dos_prueba:
        prueba.append(Elemento())
        prueba[i].pieza = 'dos'
        prueba[i].image, prueba[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Dos OK")
    
    # Analisis del numero tres en la base de datos
    iter = 0
    for objeto in tres_prueba:
        prueba.append(Elemento())
        prueba[i].pieza = 'tres'
        prueba[i].image, prueba[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Tres OK")
    
    # Analisis del numero cuatro en base de datos
    iter = 0
    for objeto in cuatro_prueba:
        prueba.append(Elemento())
        prueba[i].pieza = 'cuatro'
        prueba[i].image, prueba[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Cuatro OK")

    # Analisis del numero cinco en base de datos
    iter = 0
    for objeto in cinco_prueba:
        prueba.append(Elemento())
        prueba[i].pieza = 'cinco'
        prueba[i].image, prueba[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Cinco OK")

    # Analisis del numero seis en la base de datos
    iter = 0
    for objeto in seis_prueba:
        prueba.append(Elemento())
        prueba[i].pieza = 'seis'
        prueba[i].image, prueba[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Seis OK")
    
    # Analisis del numero siete en la base de datos
    iter = 0
    for objeto in siete_prueba:
        prueba.append(Elemento())
        prueba[i].pieza = 'siete'
        prueba[i].image, prueba[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Siete OK")

    # Analisis del numero ocho en la base de datos
    iter = 0
    for objeto in ocho_prueba:
        prueba.append(Elemento())
        prueba[i].pieza = 'ocho'
        prueba[i].image, prueba[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Ocho OK")
    
    # Analisis del numero nueve en la base de datos
    iter = 0
    for objeto in nueve_prueba:
        prueba.append(Elemento())
        prueba[i].pieza = 'nueve'
        prueba[i].image, prueba[i].caracteristica = extraccion(objeto)
        i += 1
        iter += 1
    print("Nueve OK")

    print("Testeo de todos los objetos en Test completo")
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

        eval = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        for i in range(0, k):

            if (datos[i].pieza == 'punto'):
                eval[0] += 10

            if (datos[i].pieza == 'uno'):
                eval[1] += 10

            if (datos[i].pieza == 'dos'):
                eval[2] += 10
                
            if (datos[i].pieza == 'tres'):
                eval[3] += 10
            
            if (datos[i].pieza == 'cuatro'):
                eval[4] += 10

            if (datos[i].pieza == 'cinco'):
                eval[5] += 10

            if (datos[i].pieza == 'seis'):
                eval[6] += 10
                
            if (datos[i].pieza == 'siete'):
                eval[7] += 10
                
            if (datos[i].pieza == 'ocho'):
                eval[8] += 10
                
            if (datos[i].pieza == 'nueve'):
                eval[9] += 10

        aux = eval[0]
        if (aux < eval[1]):
            aux = eval[1]
        if (aux < eval[2]):
            aux = eval[2]
        if (aux < eval[3]):
            aux = eval[3]
        if (aux < eval[4]):
            aux = eval[4]
        if (aux < eval[5]):
            aux = eval[5]
        if (aux < eval[6]):
            aux = eval[6]
        if (aux < eval[7]):
            aux = eval[7]
        if (aux < eval[8]):
            aux = eval[8]
        if (aux < eval[9]):
            aux = eval[9]
            

        if (aux == eval[0]):
            pieza = 'punto'
        if (aux == eval[1]):
            pieza = 'uno'
        if (aux == eval[2]):
            pieza = 'dos'
        if (aux == eval[3]):
            pieza = 'tres'
        if (aux == eval[4]):
            pieza = 'cuatro'
        if (aux == eval[5]):
            pieza = 'cinco'
        if (aux == eval[6]):
            pieza = 'seis'
        if (aux == eval[7]):
            pieza = 'siete'
        if (aux == eval[8]):
            pieza = 'ocho'
        if (aux == eval[9]):
            pieza = 'nueve'
        
        if (t.pieza == pieza):
            correct += 1
         
    return correct

##Rendimiento KNN - Maldicion de dimensionalidad

print("Inicializacion KKN\n")
datos = analisis_de_datos()
prueba = analisis_de_prueba()

MAX = 20
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

    punto_datos = []
    uno_datos = []
    dos_datos = []
    tres_datos = []
    cuatro_datos = []
    cinco_datos = []
    seis_datos = []
    siete_datos = []
    ocho_datos = []
    nueve_datos = []
    
    # MEANS INICIALES
    for element in datos:
        if (element.pieza == 'punto'):
            punto_datos.append(element)
        if (element.pieza == 'uno'):
            uno_datos.append(element)
        if (element.pieza == 'dos'):
            dos_datos.append(element)
        if (element.pieza == 'tres'):
            tres_datos.append(element)
        if (element.pieza == 'cuatro'):
            cuatro_datos.append(element)
        if (element.pieza == 'cinco'):
            cinco_datos.append(element)
        if (element.pieza == 'seis'):
            seis_datos.append(element)
        if (element.pieza == 'siete'):
            siete_datos.append(element)
        if (element.pieza == 'ocho'):
            ocho_datos.append(element)
        if (element.pieza == 'nueve'):
            nueve_datos.append(element)

    punto_mean = list(random.choice(punto_datos).caracteristica)
    uno_mean = list(random.choice(uno_datos).caracteristica)
    dos_mean = list(random.choice(dos_datos).caracteristica)
    tres_mean = list(random.choice(tres_datos).caracteristica)
    cuatro_mean = list(random.choice(cuatro_datos).caracteristica)
    cinco_mean = list(random.choice(cinco_datos).caracteristica)
    seis_mean = list(random.choice(seis_datos).caracteristica)
    siete_mean = list(random.choice(siete_datos).caracteristica)
    ocho_mean = list(random.choice(ocho_datos).caracteristica)
    nueve_mean = list(random.choice(nueve_datos).caracteristica)

    punto_flag = True
    uno_flag = True
    dos_flag = True
    tres_flag = True
    cuatro_flag = True
    cinco_flag = True
    seis_flag = True
    siete_flag = True
    ocho_flag = True
    nueve_flag = True

    punto_len = [0, 0, 0]
    uno_len = [0, 0, 0]
    dos_len = [0, 0, 0]
    tres_len = [0, 0, 0]
    cuatro_len = [0, 0, 0]
    cinco_len = [0, 0, 0]
    seis_len = [0, 0, 0]
    siete_len = [0, 0, 0]
    ocho_len = [0, 0, 0]
    nueve_len = [0, 0, 0]

    iter = 0
    while (iter < 20):

        punto_datos = []
        uno_datos = []
        dos_datos = []
        tres_datos = []
        cuatro_datos = []
        cinco_datos = []
        seis_datos = []
        siete_datos = []
        ocho_datos = []
        nueve_datos = []
        
        # ASIGNACION
        for element in datos:
            sum_punto = 0
            sum_uno = 0
            sum_dos = 0
            sum_tres = 0
            sum_cuatro = 0
            sum_cinco = 0
            sum_seis = 0
            sum_siete = 0
            sum_ocho = 0
            sum_nueve = 0
            
            for i in range(0, len(element.caracteristica)-1):
                sum_punto += np.power(np.abs(punto_mean[i] - element.caracteristica[i]), 2)
                sum_uno += np.power(np.abs(uno_mean[i] - element.caracteristica[i]), 2)
                sum_dos += np.power(np.abs(dos_mean[i] - element.caracteristica[i]), 2)
                sum_tres += np.power(np.abs(tres_mean[i] - element.caracteristica[i]), 2)
                sum_cuatro += np.power(np.abs(cuatro_mean[i] - element.caracteristica[i]), 2)
                sum_cinco += np.power(np.abs(cinco_mean[i] - element.caracteristica[i]), 2)
                sum_seis += np.power(np.abs(seis_mean[i] - element.caracteristica[i]), 2)
                sum_siete += np.power(np.abs(siete_mean[i] - element.caracteristica[i]), 2)
                sum_ocho += np.power(np.abs(ocho_mean[i] - element.caracteristica[i]), 2)
                sum_nueve += np.power(np.abs(nueve_mean[i] - element.caracteristica[i]), 2)
                
            dist_punto = np.sqrt(sum_punto)
            dist_uno = np.sqrt(sum_uno)
            dist_dos = np.sqrt(sum_dos)
            dist_tres = np.sqrt(sum_tres)
            dist_cuatro = np.sqrt(sum_cuatro)
            dist_cinco = np.sqrt(sum_cinco)
            dist_seis = np.sqrt(sum_seis)
            dist_siete = np.sqrt(sum_siete)
            dist_ocho = np.sqrt(sum_ocho)
            dist_nueve = np.sqrt(sum_nueve)
            
            aux = dist_punto
            if (dist_uno < aux):
                aux = dist_uno
            if (dist_dos < aux):
                aux = dist_dos
            if (dist_tres < aux):
                aux = dist_tres
            if (dist_cuatro < aux):
                aux = dist_cuatro
            if (dist_cinco < aux):
                aux = dist_cinco
            if (dist_seis < aux):
                aux = dist_seis
            if (dist_siete < aux):
                aux = dist_siete
            if (dist_ocho < aux):
                aux = dist_ocho
            if (dist_nueve < aux):
                aux = dist_nueve

            if (aux == dist_punto):
                punto_datos.append(element.caracteristica)
            elif (aux == dist_uno):
                uno_datos.append(element.caracteristica)
            elif(aux == dist_dos):
                dos_datos.append(element.caracteristica)
            elif(aux == dist_tres):
                tres_datos.append(element.caracteristica)
            elif (aux == dist_cuatro):
                cuatro_datos.append(element.caracteristica)
            elif(aux == dist_cinco):
                cinco_datos.append(element.caracteristica)
            elif(aux == dist_seis):
                seis_datos.append(element.caracteristica)
            elif (aux == dist_siete):
                siete_datos.append(element.caracteristica)
            elif(aux == dist_ocho):
                ocho_datos.append(element.caracteristica)
            elif(aux == dist_nueve):
                nueve_datos.append(element.caracteristica)

        # ACTUALIZACION
        sum_punto = [0, 0, 0]
        for obj0 in punto_datos:
            sum_punto[0] += obj0[0]
            sum_punto[1] += obj0[1]
            sum_punto[2] += obj0[2]
        
        sum_uno = [0, 0, 0]
        for obj1 in uno_datos:
            sum_uno[0] += obj1[0]
            sum_uno[1] += obj1[1]
            sum_uno[2] += obj1[2]

        sum_dos = [0, 0, 0]
        for obj2 in dos_datos:
            sum_dos[0] += obj2[0]
            sum_dos[1] += obj2[1]
            sum_dos[2] += obj2[2]

        sum_tres = [0, 0, 0]
        for obj3 in tres_datos:
            sum_tres[0] += obj3[0]
            sum_tres[1] += obj3[1]
            sum_tres[2] += obj3[2]
            
        sum_cuatro = [0, 0, 0]
        for obj4 in cuatro_datos:
            sum_cuatro[0] += obj4[0]
            sum_cuatro[1] += obj4[1]
            sum_cuatro[2] += obj4[2]
            
        sum_cinco = [0, 0, 0]
        for obj5 in cinco_datos:
            sum_cinco[0] += obj5[0]
            sum_cinco[1] += obj5[1]
            sum_cinco[2] += obj5[2]

        sum_seis = [0, 0, 0]
        for obj6 in seis_datos:
            sum_seis[0] += obj6[0]
            sum_seis[1] += obj6[1]
            sum_seis[2] += obj6[2]

        sum_siete = [0, 0, 0]
        for obj7 in siete_datos:
            sum_siete[0] += obj7[0]
            sum_siete[1] += obj7[1]
            sum_siete[2] += obj7[2]
            
        sum_ocho = [0, 0, 0]
        for obj8 in ocho_datos:
            sum_ocho[0] += obj8[0]
            sum_ocho[1] += obj8[1]
            sum_ocho[2] += obj8[2]

        sum_nueve = [0, 0, 0]
        for obj9 in nueve_datos:
            sum_nueve[0] += obj9[0]
            sum_nueve[1] += obj9[1]
            sum_nueve[2] += obj9[2]

        punto_mean[0] = sum_punto[0] / len(punto_datos)
        punto_mean[1] = sum_punto[1] / len(punto_datos)
        punto_mean[2] = sum_punto[2] / len(punto_datos)

        uno_mean[0] = sum_uno[0] / len(uno_datos)
        uno_mean[1] = sum_uno[1] / len(uno_datos)
        uno_mean[2] = sum_uno[2] / len(uno_datos)

        dos_mean[0] = sum_dos[0] / len(dos_datos)
        dos_mean[1] = sum_dos[1] / len(dos_datos)
        dos_mean[2] = sum_dos[2] / len(dos_datos) 
        
        tres_mean[0] = sum_tres[0] / len(tres_datos)
        tres_mean[1] = sum_tres[1] / len(tres_datos)
        tres_mean[2] = sum_tres[2] / len(tres_datos) 
        
        cuatro_mean[0] = sum_cuatro[0] / len(cuatro_datos)
        cuatro_mean[1] = sum_cuatro[1] / len(cuatro_datos)
        cuatro_mean[2] = sum_cuatro[2] / len(cuatro_datos)

        cinco_mean[0] = sum_cinco[0] / len(cinco_datos)
        cinco_mean[1] = sum_cinco[1] / len(cinco_datos)
        cinco_mean[2] = sum_cinco[2] / len(cinco_datos)

        seis_mean[0] = sum_seis[0] / len(seis_datos)
        seis_mean[1] = sum_seis[1] / len(seis_datos)
        seis_mean[2] = sum_seis[2] / len(seis_datos) 
        
        siete_mean[0] = sum_siete[0] / len(siete_datos)
        siete_mean[1] = sum_siete[1] / len(siete_datos)
        siete_mean[2] = sum_siete[2] / len(siete_datos)
        
        ocho_mean[0] = sum_ocho[0] / len(ocho_datos)
        ocho_mean[1] = sum_ocho[1] / len(ocho_datos)
        ocho_mean[2] = sum_ocho[2] / len(ocho_datos) 
        
        nueve_mean[0] = sum_nueve[0] / len(nueve_datos)
        nueve_mean[1] = sum_nueve[1] / len(nueve_datos)
        nueve_mean[2] = sum_nueve[2] / len(nueve_datos)
        
        # CONDICION DE SALIDA
        if (punto_mean == punto_len):
            punto_flag = False
        else:
            punto_len = punto_mean

        if (uno_mean == uno_len):
            uno_flag = False
        else:
            uno_len = uno_mean

        if (dos_mean == dos_len):
            dos_flag = False
        else:
            dos_len = dos_mean
            
        if (tres_mean == tres_len):
            tres_flag = False
        else:
            tres_len = tres_mean

        if (cuatro_mean == cuatro_len):
            cuatro_flag = False
        else:
            cuatro_len = cuatro_mean

        if (cinco_mean == cinco_len):
            cinco_flag = False
        else:
            cinco_len = cinco_mean

        if (seis_mean == seis_len):
            seis_flag = False
        else:
            seis_len = seis_mean
            
        if (siete_mean == siete_len):
            siete_flag = False
        else:
            siete_len = siete_mean
            
        if (ocho_mean == ocho_len):
            ocho_flag = False
        else:
            ocho_len = ocho_mean
            
        if (nueve_mean == nueve_len):
            nueve_flag = False
        else:
            nueve_len = nueve_mean

        iter += 1
        
    return [punto_mean, uno_mean, dos_mean, tres_mean, cuatro_mean, cinco_mean, seis_mean, siete_mean, ocho_mean, nueve_mean]

#Testeo de KMeans (YTest)
def kmeans(prueba, means):
    
    punto_mean = means[0]
    uno_mean = means[1]
    dos_mean = means[2]
    tres_mean = means[3]
    cuatro_mean = means[4]
    cinco_mean = means[5]
    seis_mean = means[6]
    siete_mean = means[7]
    ocho_mean = means[8]
    nueve_mean = means[9]
    
    correct = 0

    for t in prueba:

        sum_punto = 0
        sum_uno = 0
        sum_dos = 0
        sum_tres = 0
        sum_cuatro = 0
        sum_cinco = 0
        sum_seis = 0
        sum_siete = 0
        sum_ocho = 0
        sum_nueve = 0
        
        for i in range(0, len(t.caracteristica)-1):
            sum_punto += np.power(np.abs(t.caracteristica[i] - punto_mean[i]), 2)
            sum_uno += np.power(np.abs(t.caracteristica[i] - uno_mean[i]), 2)
            sum_dos += np.power(np.abs(t.caracteristica[i] - dos_mean[i]), 2)
            sum_tres += np.power(np.abs(t.caracteristica[i] - tres_mean[i]), 2)
            sum_cuatro += np.power(np.abs(t.caracteristica[i] - cuatro_mean[i]), 2)
            sum_cinco += np.power(np.abs(t.caracteristica[i] - cinco_mean[i]), 2)
            sum_seis += np.power(np.abs(t.caracteristica[i] - seis_mean[i]), 2)
            sum_siete += np.power(np.abs(t.caracteristica[i] - siete_mean[i]), 2)
            sum_ocho += np.power(np.abs(t.caracteristica[i] - ocho_mean[i]), 2)
            sum_nueve += np.power(np.abs(t.caracteristica[i] - nueve_mean[i]), 2)

        dist_punto = np.sqrt(sum_punto)
        dist_uno = np.sqrt(sum_uno)
        dist_dos = np.sqrt(sum_dos)
        dist_tres = np.sqrt(sum_tres)
        dist_cuatro = np.sqrt(sum_cuatro)
        dist_cinco = np.sqrt(sum_cinco)
        dist_seis = np.sqrt(sum_seis)
        dist_siete = np.sqrt(sum_siete)
        dist_ocho = np.sqrt(sum_ocho)
        dist_nueve = np.sqrt(sum_nueve)
        
        aux = dist_punto
        if (dist_uno < aux):
            aux = dist_uno
        if (dist_dos < aux):
            aux = dist_dos
        if (dist_tres < aux):
            aux = dist_tres
        if (dist_cuatro < aux):
            aux = dist_cuatro
        if (dist_cinco < aux):
            aux = dist_cinco
        if (dist_seis < aux):
            aux = dist_seis
        if (dist_siete < aux):
            aux = dist_siete
        if (dist_ocho < aux):
            aux = dist_ocho
        if (dist_nueve < aux):
            aux = dist_nueve

        if (aux == dist_punto):
            pieza = 'punto'
        if (aux == dist_uno):
            pieza = 'uno'
        if (aux == dist_dos):
            pieza = 'dos'
        if (aux == dist_tres):
            pieza = 'tres'
        if (aux == dist_cuatro):
            pieza = 'cuatro'
        if (aux == dist_cinco):
            pieza = 'cinco'
        if (aux == dist_seis):
            pieza = 'seis'
        if (aux == dist_siete):
            pieza = 'siete'
        if (aux == dist_ocho):
            pieza = 'ocho'
        if (aux == dist_nueve):
            pieza = 'nueve'
        
        if (t.pieza == pieza):
            correct += 1
    
    return correct

##RENDIMIENTO
print("\nInicializacion de KMeans\n")
datos = analisis_de_datos()
prueba = analisis_de_prueba()

means = entrenamiento_kmeans(datos)

#Por mas que varies MAX siempre deberias obtener una linea recta horizontal
MAX = 20

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

 
 
