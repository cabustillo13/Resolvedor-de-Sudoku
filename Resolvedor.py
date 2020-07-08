#Resolvedor de SUDOKU 3X3
#Modificado por: Carlos Bustillo
#Bibliografia:https://github.com/jorditorresBCN/Sudoku/blob/master/Sudoku.ipynb

rows = 'ABCDEFGHI'
cols = '123456789'

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a+b for a in A for b in B]

boxes = cross(rows, cols)
row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]

unitlist = row_units + column_units + square_units

units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)

#Para mostrar la grilla
def display(values):  
    width = 1+max(len(values[s]) for s in boxes)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|' if c in '36' else '')
                      for c in cols))
        if r in 'CF': print(line)
    return

def grid_values_original(grid):
    return dict(zip(boxes, grid))

def grid_values(grid):
    values = []
    for c in grid:
        if c == '.':
            values.append('123456789')
        elif c in '123456789':
            values.append(c)
    return dict(zip(boxes, values))

#Elimina los valores que ya estan dentro de la grilla
def eliminate(values):
    """Eliminate values from peers of each box with a single value.
    """
    
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            values[peer] = values[peer].replace(digit,'')
    return values

def only_choice(values):
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                values[dplaces[0]] = digit
    return values

def reduce_sudoku(values):
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])

        # se the Eliminate Strategy
        values = eliminate(values)

        # Use the Only Choice Strategy
        values = only_choice(values)

        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values

def search(values):
    values = reduce_sudoku(values)
    if values is False:
        return False ## Failed earlier
    if all(len(values[s]) == 1 for s in boxes): 
        return values ## Solved!
    
    # Choose one of the unfilled squares with the fewest possibilities
    unfilled_squares= [(len(values[s]), s) for s in boxes if len(values[s]) > 1]
    n,s = min(unfilled_squares)
    
    # recurrence to solve each one of the resulting sudokus
    for value in values[s]:
        nova_sudoku = values.copy()
        nova_sudoku[s] = value
        attempt = search(nova_sudoku)
        if attempt:
            return attempt

def solve(grid):
    # Create a dictionary of values from the grid
    values = grid_values(grid)
    return search(values)

#Main
if __name__ == '__main__':
    
    #Con manejador de archivos
    archivo = open("vector.txt","r")
    lineas = archivo.read()
    #lineas = lineas.replace('\r', '').replace('\n', '')
    archivo.close() 
    
    print ("original:")
    display(grid_values_original(lineas))
    print (" ")
    print ("solucion:")
    display(solve(lineas))

