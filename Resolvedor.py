#Resolvedor de SUDOKU 3X3
#Modificado por: Carlos Bustillo
#Bibliografia:https://github.com/jorditorresBCN/Sudoku/blob/master/Sudoku.ipynb

# Diccionario con los numeros de la grilla
def resolver(grilla):
    
    valores = valoresGrilla(grilla)
    return buscar(valores)

# Intercambio de elementos
def intercambiar(A, B):
    
    return [a+b for a in A for b in B]

# Para mostrar la grilla
def mostrar(numeros):  
    ancho = 1+max(len(numeros[s]) for s in casillas)
    line = '+'.join(['-'*(ancho*3)]*3)
    for r in filas:
        print(''.join(numeros[r+c].center(ancho)+('|' if c in '36' else '')
                      for c in columnas))
        if r in 'CF': print(line)

# Definir valores iniciales
def valoresIniciales(grilla):
    return dict(zip(casillas, grilla))

# Definir valores en la grilla
def valoresGrilla(grilla):
    numeros = []
    for c in grilla:
        if c == '.':
            numeros.append('123456789')
        elif c in '123456789':
            numeros.append(c)
    return dict(zip(casillas, numeros))

# Elimina los valores que ya estan dentro de la grilla
def eliminar(numeros):
    
    solved_values = [box for box in numeros.keys() if len(numeros[box]) == 1]
    for box in solved_values:
        digit = numeros[box]
        for vecino in vecinos[box]:
            numeros[vecino] = numeros[vecino].replace(digit,'')
    return numeros

def unica_opcion(numeros):
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in numeros[box]]
            if len(dplaces) == 1:
                numeros[dplaces[0]] = digit
    return numeros

def reduce_sudoku(numeros):
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in numeros.keys() if len(numeros[box]) == 1])

        # se the Eliminate Strategy
        numeros = eliminar(numeros)

        # Use the Only Choice Strategy
        numeros = unica_opcion(numeros)

        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in numeros.keys() if len(numeros[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in numeros.keys() if len(numeros[box]) == 0]):
            return False
    return numeros

def buscar(numeros):
    numeros = reduce_sudoku(numeros)
    if numeros is False:
        return False    ##Fallo
    if all(len(numeros[s]) == 1 for s in casillas): 
        return numeros  ## Listo
    
    # Choose one of the unfilled boxes
    unfilled_squares= [(len(numeros[s]), s) for s in casillas if len(numeros[s]) > 1]
    n,s = min(unfilled_squares)
    
    # Solve the next boxes
    for value in numeros[s]:
        nova_sudoku = numeros.copy()
        nova_sudoku[s] = value
        attempt = buscar(nova_sudoku)
        if attempt:
            return attempt

filas = 'ABCDEFGHI'
columnas = '123456789'

casillas = intercambiar(filas, columnas)
fila_units = [intercambiar(r, columnas) for r in filas]
columna_units = [intercambiar(filas, c) for c in columnas]
cuadrado_units = [intercambiar(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]

unitlist = fila_units + columna_units + cuadrado_units

unidades = dict((s, [u for u in unitlist if s in u]) for s in casillas)
vecinos = dict((s, set(sum(unidades[s],[]))-set([s])) for s in casillas)

###########################################
####                MAIN               ####
###########################################

if __name__ == '__main__':
    
    #Con manejador de archivos
    archivo = open("vector.txt","r")
    lineas = archivo.read()
    #lineas = lineas.replace('\r', '').replace('\n', '')
    archivo.close() 
    
    print ("Original:")
    mostrar(valoresIniciales(lineas))
    print (" ")
    print ("Solucion:")
    mostrar(resolver(lineas))
