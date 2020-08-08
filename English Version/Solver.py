import numpy as np

#Dictionary with grid numbers
def solverGrid(grid):
    
    values = valuesGrid(grid)
    return searchValues(values)

#Exchange of items
def exchangeValues(A, B):
    
    return [a+b for a in A for b in B]

#Define initial values
def initialValues(grid):
    return dict(zip(sections, grid))

#Define values in the grid
def valuesGrid(grid):
    numbers = []
    for c in grid:
        if c == '.':
            numbers.append('123456789')
        elif c in '123456789':
            numbers.append(c)
    return dict(zip(sections, numbers))

#Delete the values that are already inside the grid
def eliminateValues(numbers):
    
    solved_values = [box for box in numbers.keys() if len(numbers[box]) == 1]
    for box in solved_values:
        digit = numbers[box]
        for vecino in neighbors[box]:
            numbers[vecino] = numbers[vecino].replace(digit,'')
    return numbers

def onlyOption(numbers):
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in numbers[box]]
            if len(dplaces) == 1:
                numbers[dplaces[0]] = digit
    return numbers

def reduceSudoku(numbers):
    stalled = False
    while not stalled:
        #Check how many boxes have a determined value
        solved_values_before = len([box for box in numbers.keys() if len(numbers[box]) == 1])

        #Set the Eliminate Strategy
        numbers = eliminateValues(numbers)

        #Use the Only Choice Strategy
        numbers = onlyOption(numbers)

        #Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in numbers.keys() if len(numbers[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        #Sanity check, return False if there is a box with zero available values:
        if len([box for box in numbers.keys() if len(numbers[box]) == 0]):
            return False
    return numbers

def searchValues(numbers):
    numbers = reduceSudoku(numbers)
    if numbers is False:
        return False    ##Failure
    if all(len(numbers[s]) == 1 for s in sections): 
        return numbers  ##Ok
    
    #Choose one of the unfilled boxes
    unfilled_squares= [(len(numbers[s]), s) for s in sections if len(numbers[s]) > 1]
    n,s = min(unfilled_squares)
    
    #Solve the next boxes
    for value in numbers[s]:
        nova_sudoku = numbers.copy()
        nova_sudoku[s] = value
        attempt = searchValues(nova_sudoku)
        if attempt:
            return attempt

#Define values
rows = 'ABCDEFGHI'
columns = '123456789'

sections = exchangeValues(rows, columns)
rowsUnit = [exchangeValues(r, columns) for r in rows]
columnUnits = [exchangeValues(rows, c) for c in columns]
boxUnits = [exchangeValues(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]

unitlist = rowsUnit + columnUnits + boxUnits

units = dict((s, [u for u in unitlist if s in u]) for s in sections)
neighbors = dict((s, set(sum(units[s],[]))-set([s])) for s in sections)

#MAIN
if __name__ == '__main__':
    
    #With file manager to read the file vector.txt that has all the values of the screenshot
    file = open("vector.txt","r")
    lines = file.read()
    file.close() 

    #Access the dictionary
    a = solverGrid(lines)
    b = sorted(a.items())
    # Save the dictionary solution
    np.save('Solution', b) 
