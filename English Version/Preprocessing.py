import cv2
import numpy as np
import Functions

#Relative path
path="./Screenshots/"

#Image to analize
number = input("Enter image number: ")
globalPath = path+"photo"+str(number)+".png"
image = cv2.imread(globalPath)

#Save the name of the image to analize after in Main.py
file = open("image.txt","w")
file.write(globalPath)
file.close()

# MAIN
if __name__ == '__main__':    
    
    ##PREPROCESSING -> Crop the edges, ads and all the images outside the sudoku board
    image = Functions.cropImage(image,218)
    image = Functions.rotateImage(image,180)
    image = Functions.cropImage(image,348)
    image = Functions.rotateImage(image,180)
    
    ##Crop each box in the sudoku board
    cont=0
    w=0
    for j in range(9):
        h=0
        for i in range(9):
            nombre = "image"+ str(cont) + ".png"
            image1 = Functions.cropBox(image,w,h,75,80)
            #Save the image
            Functions.saveImage(image1,nombre)
            h=h+80
            cont=cont+1
        #Position of the pixel where start the image
        w=80*(j+1)
        
