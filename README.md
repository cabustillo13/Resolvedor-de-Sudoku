# Resolvedor-de-Sudoku

Programa en python 2.7 para resolver un sudoku de la app para Android "Sudoku" de genina.com.
Se analiza mediante visión artificial un screenshot al juego, se obtiene una imagen de 720x1280.

¿Cómo funciona?
1) Run Preprocesamiento.py
Lo que hace es extraer cada casilla del sudoku de manera individual y los guarda de manera secuencial como photo#.png (donde # va de 0 a 80).
Se obtienen imágenes de 80x75

2) Run Transformacion.py
Lo que hace es recortar los bordes de cada casilla, por si ha quedado algún borde negro que pueda inferir en nuestro análisis.
Se obtienen imágenes de 56x51

3) Run Main.py
LO que hace es analizar que número se encuentra en la casilla.
Para este caso se utiliza el algoritmo de Canny para determinar si hay algún número o es una casilla vacía.
Luego a través de algoritmo KNN se determina que número se encuentra en la casilla.
Para la extracción de características se utilizo los momentos de Hu: 1 y 2, filtro gaussiano para la filtración y thresholding no supervisada.

4) Run Resolvedor.py
Lo que hace es resolver el sudoku.
Se presenta un algoritmo de satisfacción de restricciones con backtracking.

Otros programas:

5) Funciones.py
Contiene todas las funciones que se utilizan para preprocesamiento y transformación de la imagen.

6) Rendimiento.py
Se plantea un algoritmo KNN y KMeans y se evaluan todas las imágenes de las carpeta Test, y así determinar su rendimiento del porcentaje de predicciones.
AL final se eligíó KNN porque presenta un rendimiento mayor.
