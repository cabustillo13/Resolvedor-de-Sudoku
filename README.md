# Resolvedor-de-Sudoku

**Este código fue publicado en:** https://medium.com/@carlosbustillo/solve-sudoku-9x9-with-computer-vision-and-a-constraint-satisfaction-algorithm-in-python-7bb27769c1eb

Programa en python 2.7 para resolver un sudoku de la app para Android "Sudoku" de genina.com.
Se toma un screenshot del juego (se obtiene una imagen de 720x1280), luego se extrae información útil a partir de visión artificial y posteriormente se analiza con un algoritmo de satisfacción de restricciones con backtracking. 

**Librerías utilizadas**

* cv2
* numpy
* matplotlib
* skimage
* mahotas

## **¿Cómo funciona?**
**1) Run Preprocesamiento.py**

Lo que hace es extraer cada casilla del sudoku de manera individual y los guarda de manera secuencial como photo#.png (donde # va de 0 a 80).
Se obtienen imágenes de 80x75

```python Preprocesamiento.py```

**2) Run Transformacion.py**

Lo que hace es recortar los bordes de cada casilla, por si ha quedado algún borde negro que pueda inferir en nuestro análisis.
Se obtienen imágenes de 56x51

```python Transformacion.py```

**3) Run Main.py**

Lo que hace es analizar que número se encuentra en la casilla.
Para este caso se utiliza el algoritmo de Canny para determinar si hay algún número o es una casilla vacía.
Luego a través de algoritmo KNN se determina que número se encuentra en la casilla.
Para la extracción de características se utilizo los momentos de Hu: 1 y 2, filtro gaussiano para la filtración y thresholding no supervisada.

```python Main.py```

**4) Run Resolvedor.py**

Lo que hace es resolver el sudoku.
Se presenta un algoritmo de satisfacción de restricciones con backtracking.
Para esta etapa tome como referencia este repositorio: https://github.com/jorditorresBCN/Sudoku

```python Resolvedor.py```

**5) Run Interfaz.py**

Mejora la forma en que se visualiza la solución respecto a la screenshot original. 

```python Interfaz.py```

## **Run programa entero en bash**

Ejecute el script play.sh en su consola, para ejecutar en un solo paso todo el programa si lo desea.
Depende de la consola que se tiene, la forma que se ejecuta.
En mi consola lo ejecuto como: *sh ./play.sh*

## **Otros programas:**

**6) Funciones.py**

Contiene todas las funciones que se utilizan para preprocesamiento y transformación de la imagen.

**7) Rendimiento.py**

Se plantea un algoritmo KNN y KMeans y se evaluan todas las imágenes de las carpeta Test, y así determinar su rendimiento del porcentaje de predicciones.
AL final se eligíó KNN porque presenta un rendimiento mayor.

```python Rendimiento.py```

**8) vector.txt**

Contiene todos los elementos extraídos del screenshot (donde se recorrio las casillas de izquierda a derecha, de arriba hacia abajo).
Según Rendimiento.py, el algoritmo KNN presento un rendimiento del 97% respecto a todas las imágenes analizadas en el Test. En caso, de algún error en el reconocimiento de los números existe la opción de cambiar manualmente una predicción de la casilla en el vector.txt.

**9) imagen.txt**

Contiene el path de la imagen que se va a analizar.

**10) Solucion.npy**

Es un diccionario que contiene la solución del sudoku. 


    

    
      

      

    
    



  







  



  

      
        

        
        

  
    
  
