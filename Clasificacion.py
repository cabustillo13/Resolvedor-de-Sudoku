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

# Elemento a evaluar
#Recordar aplicar Transformacion.py cuando se quiera evaluar una nueva imagen.
test = Elemento()

for numero in range(81):

    nombre = './Imagenes/prueba'+str(numero)+'.png'
    image = io.imread(nombre)

    test.image, test.caracteristica = extraccion(image)
    test.pieza = 'Arandela' # label inicial 

    ax.scatter(test.caracteristica[0], test.caracteristica[1], test.caracteristica[2], c='k', marker='o')
    fig

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

    #print("\nPredicciones para KNN con K=3: ")            
    #k = 3
    #for i in range(k):
    #    print(datos[i].pieza)

    #K MEANS
    import random
    print("\nInicializacion KMeans")

    punto_data = []
    uno_data = []
    dos_data = []
    tres_data = []
    cuatro_data = []
    cinco_data = []
    seis_data = []
    siete_data = []
    ocho_data = []
    nueve_data = []

    for element in datos:
        if (element.pieza == 'Punto'):
            punto_data.append(element)
        if (element.pieza == 'Uno'):
            uno_data.append(element)
        if (element.pieza == 'Dos'):
            dos_data.append(element)
        if (element.pieza == 'Tres'):
            tres_data.append(element)
        if (element.pieza == 'Cuatro'):
            cuatro_data.append(element)
        if (element.pieza == 'Cinco'):
            cinco_data.append(element)
        if (element.pieza == 'Seis'):
            seis_data.append(element)
        if (element.pieza == 'Siete'):
            siete_data.append(element)
        if (element.pieza == 'Ocho'):
            ocho_data.append(element)
        if (element.pieza == 'Nueve'):
            nueve_data.append(element)
        
    punto_mean = list(random.choice(punto_data).caracteristica)
    uno_mean = list(random.choice(uno_data).caracteristica)
    dos_mean = list(random.choice(dos_data).caracteristica)
    tres_mean = list(random.choice(tres_data).caracteristica)
    cuatro_mean = list(random.choice(cuatro_data).caracteristica)
    cinco_mean = list(random.choice(cinco_data).caracteristica)
    seis_mean = list(random.choice(seis_data).caracteristica)
    siete_mean = list(random.choice(siete_data).caracteristica)
    ocho_mean = list(random.choice(ocho_data).caracteristica)
    nueve_mean = list(random.choice(nueve_data).caracteristica)

    fig_means = plt.figure()
    ax = fig_means.add_subplot(111, projection='3d')

    # fig_means, ax = plt.subplots()
    ax.scatter(punto_mean[0], punto_mean[1], punto_mean[2], c='y', marker='o')
    ax.scatter(uno_mean[0], uno_mean[1], uno_mean[2], c='r', marker='o')
    ax.scatter(dos_mean[0], dos_mean[1], dos_mean[2], c='b', marker='o')
    ax.scatter(tres_mean[0], tres_mean[1], tres_mean[2], c='g', marker='o')
    ax.scatter(cuatro_mean[0], cuatro_mean[1], cuatro_mean[2], c='c', marker='o')
    ax.scatter(cinco_mean[0], cinco_mean[1], cinco_mean[2], c='m', marker='o')
    ax.scatter(seis_mean[0], seis_mean[1], seis_mean[2], c='k', marker='o')
    ax.scatter(siete_mean[0], siete_mean[1], siete_mean[2], c='lime', marker='o')
    ax.scatter(ocho_mean[0], ocho_mean[1], ocho_mean[2], c='aqua', marker='o')
    ax.scatter(nueve_mean[0], nueve_mean[1], nueve_mean[2], c='purple', marker='o')

    ax.grid(True)
    ax.set_title("Means")

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

    plt.legend(handles=[yellow_patch, red_patch, blue_patch, green_patch, cyan_patch, magenta_patch, black_patch, lime_patch, aqua_patch, purple_patch])

    ax.set_xlabel('componente 1')
    ax.set_ylabel('componente 2')
    ax.set_zlabel('componente 4')

    plt.show()

    # Asignacion, Actualizacion y Convergencia
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

        punto_data = []
        uno_data = []
        dos_data = []
        tres_data = []
        cuatro_data = []
        cinco_data = []
        seis_data = []
        siete_data = []
        ocho_data = []
        nueve_data = []

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
                punto_data.append(element.caracteristica)
            elif (aux == dist_uno):
                uno_data.append(element.caracteristica)
            elif(aux == dist_dos):
                dos_data.append(element.caracteristica)
            elif(aux == dist_tres):
                tres_data.append(element.caracteristica)
            elif (aux == dist_cuatro):
                cuatro_data.append(element.caracteristica)
            elif(aux == dist_cinco):
                cinco_data.append(element.caracteristica)
            elif(aux == dist_seis):
                seis_data.append(element.caracteristica)
            elif (aux == dist_siete):
                siete_data.append(element.caracteristica)
            elif(aux == dist_ocho):
                ocho_data.append(element.caracteristica)
            elif(aux == dist_nueve):
                nueve_data.append(element.caracteristica)
            
        # ACTUALIZACION
        sum_punto = [0, 0, 0]
        for b in punto_data:
            sum_punto[0] += b[0]
            sum_punto[1] += b[1]
            sum_punto[2] += b[2]

        sum_uno = [0, 0, 0]
        for o in uno_data:
            sum_uno[0] += o[0]
            sum_uno[1] += o[1]
            sum_uno[2] += o[2]

        sum_dos = [0, 0, 0]
        for l in dos_data:
            sum_dos[0] += l[0]
            sum_dos[1] += l[1]
            sum_dos[2] += l[2]

        sum_tres = [0, 0, 0]
        for p in tres_data:
            sum_tres[0] += p[0]
            sum_tres[1] += p[1]
            sum_tres[2] += p[2]
        
        sum_cuatro = [0, 0, 0]
        for x1 in cuatro_data:
            sum_cuatro[0] += x1[0]
            sum_cuatro[1] += x1[1]
            sum_cuatro[2] += x1[2]

        sum_cinco = [0, 0, 0]
        for x2 in cinco_data:
            sum_cinco[0] += x2[0]
            sum_cinco[1] += x2[1]
            sum_cinco[2] += x2[2]

        sum_seis = [0, 0, 0]
        for x3 in seis_data:
            sum_seis[0] += x3[0]
            sum_seis[1] += x3[1]
            sum_seis[2] += x3[2]

        sum_siete = [0, 0, 0]
        for x4 in siete_data:
            sum_siete[0] += x4[0]
            sum_siete[1] += x4[1]
            sum_siete[2] += x4[2]
    
        sum_ocho = [0, 0, 0]
        for x5 in ocho_data:
            sum_ocho[0] += x5[0]
            sum_ocho[1] += x5[1]
            sum_ocho[2] += x5[2]

        sum_nueve = [0, 0, 0]
        for x6 in nueve_data:
            sum_nueve[0] += x6[0]
            sum_nueve[1] += x6[1]
            sum_nueve[2] += x6[2]
    
    
        punto_mean[0] = sum_punto[0] / len(punto_data)
        punto_mean[1] = sum_punto[1] / len(punto_data)
        punto_mean[2] = sum_punto[2] / len(punto_data)

        uno_mean[0] = sum_uno[0] / len(uno_data)
        uno_mean[1] = sum_uno[1] / len(uno_data)
        uno_mean[2] = sum_uno[2] / len(uno_data)

        dos_mean[0] = sum_dos[0] / len(dos_data)
        dos_mean[1] = sum_dos[1] / len(dos_data)
        dos_mean[2] = sum_dos[1] / len(dos_data)
    
        tres_mean[0] = sum_tres[0] / len(tres_data)
        tres_mean[1] = sum_tres[1] / len(tres_data)
        tres_mean[2] = sum_tres[1] / len(tres_data)
    
        cuatro_mean[0] = sum_cuatro[0] / len(cuatro_data)
        cuatro_mean[1] = sum_cuatro[1] / len(cuatro_data)
        cuatro_mean[2] = sum_cuatro[2] / len(cuatro_data)

        cinco_mean[0] = sum_cinco[0] / len(cinco_data)
        cinco_mean[1] = sum_cinco[1] / len(cinco_data)
        cinco_mean[2] = sum_cinco[2] / len(cinco_data)

        seis_mean[0] = sum_seis[0] / len(seis_data)
        seis_mean[1] = sum_seis[1] / len(seis_data)
        seis_mean[2] = sum_seis[1] / len(seis_data)
    
        siete_mean[0] = sum_siete[0] / len(siete_data)
        siete_mean[1] = sum_siete[1] / len(siete_data)
        siete_mean[2] = sum_siete[1] / len(siete_data)
    
        ocho_mean[0] = sum_ocho[0] / len(ocho_data)
        ocho_mean[1] = sum_ocho[1] / len(ocho_data)
        ocho_mean[2] = sum_ocho[1] / len(ocho_data)
    
        nueve_mean[0] = sum_nueve[0] / len(nueve_data)
        nueve_mean[1] = sum_nueve[1] / len(nueve_data)
        nueve_mean[2] = sum_nueve[1] / len(nueve_data)
    
        #print("Punto Uno Dos Tres Cuatro Cinco Seis Siete Ocho Nueve")
        #print(len(punto_data), len(uno_data), len(dos_data), len(tres_data), len(cuatro_data), len(cinco_data), len(seis_data), len(siete_data), len(ocho_data), len(nueve_data))
    
        # CONVERGENCIA Y CONDICION DE SALIDA
    
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
    
    # Ubicacion de los means finales
    ax.scatter(punto_mean[0], punto_mean[1], punto_mean[2], c='k', marker='o')
    ax.scatter(uno_mean[0], uno_mean[1], uno_mean[2], c='k', marker='o')
    ax.scatter(dos_mean[0], dos_mean[1], dos_mean[2], c='k', marker='o')
    ax.scatter(tres_mean[0], tres_mean[1], tres_mean[2], c='k', marker='o')
    ax.scatter(cuatro_mean[0], cuatro_mean[1], cuatro_mean[2], c='k', marker='o')
    ax.scatter(cinco_mean[0], cinco_mean[1], cinco_mean[2], c='k', marker='o')
    ax.scatter(seis_mean[0], seis_mean[1], seis_mean[2], c='k', marker='o')
    ax.scatter(siete_mean[0], siete_mean[1], siete_mean[2], c='k', marker='o')
    ax.scatter(ocho_mean[0], ocho_mean[1], ocho_mean[2], c='k', marker='o')
    ax.scatter(nueve_mean[0], nueve_mean[1], nueve_mean[2], c='k', marker='o')

    print("Ubicacion de los means finales")
    print("Punto Uno Dos Tres Cuatro Cinco Seis Siete Ocho Nueve")
    print(len(punto_data), len(uno_data), len(dos_data), len(tres_data),len(cuatro_data), len(cinco_data), len(seis_data), len(siete_data), len(ocho_data), len(nueve_data))
    fig_means

    ##Mean mas cercano
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

    for i in range(0, len(test.caracteristica)-1):
        sum_punto += np.power(np.abs(test.caracteristica[i] - punto_mean[i]), 2)
        sum_uno += np.power(np.abs(test.caracteristica[i] - uno_mean[i]), 2)
        sum_dos += np.power(np.abs(test.caracteristica[i] - dos_mean[i]), 2)
        sum_tres += np.power(np.abs(test.caracteristica[i] - tres_mean[i]), 2)
        sum_cuatro += np.power(np.abs(test.caracteristica[i] - cuatro_mean[i]), 2)
        sum_cinco += np.power(np.abs(test.caracteristica[i] - cinco_mean[i]), 2)
        sum_seis += np.power(np.abs(test.caracteristica[i] - seis_mean[i]), 2)
        sum_siete += np.power(np.abs(test.caracteristica[i] - siete_mean[i]), 2)
        sum_ocho += np.power(np.abs(test.caracteristica[i] - ocho_mean[i]), 2)
        sum_nueve += np.power(np.abs(test.caracteristica[i] - nueve_mean[i]), 2)

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

    print("\nMean mas cercano")
    print("Punto Uno Dos Tres Cuatro Cinco Seis Siete Ocho Nueve")
    print(dist_punto, dist_uno, dist_dos, dist_tres, dist_cuatro, dist_cinco, dist_seis, dist_siete, dist_ocho, dist_nueve)

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
        test.pieza = 'Punto'
    elif (aux == dist_uno):
        test.pieza = 'Uno'
    elif(aux == dist_dos):
        test.pieza = 'Dos'
    elif(aux == dist_tres):
        test.pieza = 'Tres'
    elif (aux == dist_cuatro):
        test.pieza = 'Cuatro'
    elif(aux == dist_cinco):
        test.pieza = 'Cinco'
    elif(aux == dist_seis):
        test.pieza = 'Seis'
    elif (aux == dist_siete):
        test.pieza = 'Siete'
    elif(aux == dist_ocho):
        test.pieza = 'Ocho'
    elif(aux == dist_nueve):
        test.pieza = 'Nueve'

    print("\nPrediccion para KMeans: ")
    print(test.pieza)
