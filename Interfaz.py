import numpy as np
import matplotlib.pyplot as plt
import cv2

# Leer diccionario desde el archivo .npy
leerDiccionario = np.load('Solucion.npy')
valores = (leerDiccionario[:,1])

# Leer vector.txt
archivo = open("vector.txt","r")
lineas = archivo.read()
archivo.close()

# Leer la ruta de la imagen que buscamos
archivo = open("imagen.txt","r")
pathGlobal = archivo.read()
archivo.close()

# Obtener las coordenadas para poder ubicarlas en la imagen
fila = ["A","B","C","D","E","F","G","H","I"]
columna = ["1","2","3","4","5","6","7","8","9"]

# Asignar las coordenadas de cada numero dentro del plano de la imagen
# Las coordenas fueron determinadas con ayuda de photopea
def coordenada():
    posicionx = list()
    posiciony = list()
    
    for k in range(9):
        for i in range(9):
        
            if (fila[k] == "A"): y = 270 
            elif (fila[k] == "B"): y = 350
            elif (fila[k] == "C"): y = 430
            elif (fila[k] == "D"): y = 510
            elif (fila[k] == "E"): y = 590
            elif (fila[k] == "F"): y = 670
            elif (fila[k] == "G"): y = 750
            elif (fila[k] == "H"): y = 830
            elif (fila[k] == "I"): y = 915
        
            if (columna[i] == "1"): x = 19
            elif (columna[i] == "2"): x = 98
            elif (columna[i] == "3"): x = 182
            elif (columna[i] == "4"): x = 261
            elif (columna[i] == "5"): x = 335
            elif (columna[i] == "6"): x = 419
            elif (columna[i] == "7"): x = 499
            elif (columna[i] == "8"): x = 580
            elif (columna[i] == "9"): x = 660
        
            #print (fila[k]+columna[i])
            posicionx.append(x)
            posiciony.append(y)
        
    return (posicionx,posiciony)        

# Permite escibir el valor dentro de la imagen
def escribirValor(image,valor,x,y):
        
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    texto = str(valor)
    
    # Escribir texto en la imagen
    cv2.putText(image, texto, (x,y), fuente, 2, (255, 0, 0), 5)
    #cv2.putText(imagen,texto, (coordenadas),tamano fuente,(color RGB),grosor)
    
    return image

# Cargar imagen
image = cv2.imread(pathGlobal)
image2 = image.copy()

# Cargar coordenadas
posicionx, posiciony = coordenada()

for i in range(81):
    if (lineas[i] == "."):
        image = escribirValor(image,valores[i],posicionx[i],posiciony[i])

# Unir imagenes horizontalmente
image = np.concatenate((image2,image),axis = 1)

# Mostrar concatenacion de imagenes    
plt.imshow(image)
plt.axis("off")
plt.show()

# Guardar imagenes
cv2.imwrite("./Interfaz/ejemplo.png",image)
