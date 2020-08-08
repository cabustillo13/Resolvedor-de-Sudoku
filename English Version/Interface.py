import numpy as np
import matplotlib.pyplot as plt
import cv2

#Read dictionary from Solution.npy
readDictionary = np.load('Solution.npy')
values = (readDictionary[:,1])

#Read vector.txt
file = open("vector.txt","r")
lines = file.read()
file.close()

# Read the path of the image the we want to analize
fileTxt = open("image.txt","r")
pathGlobal = fileTxt.read()
fileTxt.close()

#Obtain the coordinates to be able to locate them in the image
row = ["A","B","C","D","E","F","G","H","I"]
column = ["1","2","3","4","5","6","7","8","9"]

#Assign the coordinates of each number within the image plane
def coordinate():
    positionx = list()
    positiony = list()
    
    for k in range(9):
        for i in range(9):
        
            if (row[k] == "A"): y = 270 
            elif (row[k] == "B"): y = 350
            elif (row[k] == "C"): y = 430
            elif (row[k] == "D"): y = 510
            elif (row[k] == "E"): y = 590
            elif (row[k] == "F"): y = 670
            elif (row[k] == "G"): y = 750
            elif (row[k] == "H"): y = 830
            elif (row[k] == "I"): y = 915
        
            if (column[i] == "1"): x = 19
            elif (column[i] == "2"): x = 98
            elif (column[i] == "3"): x = 182
            elif (column[i] == "4"): x = 261
            elif (column[i] == "5"): x = 335
            elif (column[i] == "6"): x = 419
            elif (column[i] == "7"): x = 499
            elif (column[i] == "8"): x = 580
            elif (column[i] == "9"): x = 660

            positionx.append(x)
            positiony.append(y)
        
    return (positionx,positiony)        

#Function to write value in each box in the image
def writeValue(image,valor,x,y):
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = str(valor)
    
    #Write text in the image
    cv2.putText(image, text, (x,y), font, 2, (255, 0, 0), 5)
    #cv2.putText(image,text, (coordinates),size font,(color RGB),thickness)
    
    return image

#Load image
image = cv2.imread(pathGlobal)
image2 = image.copy()

#Load coordinates
positionx, positiony = coordinate()

for i in range(81):
    if (lines[i] == "."):
        image = writeValue(image,values[i],positionx[i],positiony[i])

# Concatenate images horizontally
image = np.concatenate((image2,image),axis = 1)

#Show image concatenation   
plt.imshow(image)
plt.axis("off")
plt.show()

#Save image
cv2.imwrite("./Interface/example.png",image)
