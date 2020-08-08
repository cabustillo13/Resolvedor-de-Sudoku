import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams['image.cmap'] = 'gray'
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, color, img_as_float, filters
from skimage.feature import hog
import cv2
import mahotas
    
# Function to extract characteristics of the images to later use them in the knn algorithm
def extraction(image):
    
    ##PREPROCESSING -> Convert image to grayscale
    aux = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    ##FILTERING -> Apply Gauss Filter
    aux = cv2.GaussianBlur(aux, (3, 3), 0)   
    ##SEGMENTATION -> Apply Thresholding simple
    ret, th = cv2.threshold(aux, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    aux= th
    ##FEATURE EXTRACTION -> Obtain Hu Moments
    hu = cv2.HuMoments(cv2.moments(aux)).flatten()
    ##Analysis the features (Hu Moments)
    return aux, [hu[0], hu[1]]

#Training Data Base (YTrain)
##Load all images of each numbers that appears in sudoku board 
number1 = io.ImageCollection('./Images/Train/Y1/*.png:./Images/Train/Y1/*.jpg')
number2 = io.ImageCollection('./Images/Train/Y2/*.png:./Images/Train/Y2/*.jpg')
number3 = io.ImageCollection('./Images/Train/Y3/*.png:./Images/Train/Y3/*.jpg')
number4 = io.ImageCollection('./Images/Train/Y4/*.png:./Images/Train/Y4/*.jpg')
number5 = io.ImageCollection('./Images/Train/Y5/*.png:./Images/Train/Y5/*.jpg')
number6 = io.ImageCollection('./Images/Train/Y6/*.png:./Images/Train/Y6/*.jpg')
number7 = io.ImageCollection('./Images/Train/Y7/*.png:./Images/Train/Y7/*.jpg')
number8 = io.ImageCollection('./Images/Train/Y8/*.png:./Images/Train/Y8/*.jpg')
number9 = io.ImageCollection('./Images/Train/Y9/*.png:./Images/Train/Y9/*.jpg')
    
#Create a class for each element
class Element:
    def __init__(self):
        self.number = None
        self.image = None
        self.feature = []
        self.distance = 0
        
#Analize data
data = []
i = 0

#Analize number 1
iter = 0
for object in number1:
    data.append(Element())
    data[i].number = '1'
    data[i].image, data[i].feature = extraction(object)
    i += 1
    iter += 1
print("number1 is OK")

#Analize number 2
iter = 0
for object in number2:
    data.append(Element())
    data[i].number = '2'
    data[i].image, data[i].feature = extraction(object)
    i += 1
    iter += 1
print("number2 is OK")

#Analize number 3
iter = 0
for object in number3:
    data.append(Element())
    data[i].number = '3'
    data[i].image, data[i].feature = extraction(object)
    i += 1
    iter += 1
print("number3 is OK")

#Analize number 4
iter = 0
for object in number4:
    data.append(Element())
    data[i].number = '4'
    data[i].image, data[i].feature = extraction(object)
    i += 1
    iter += 1
print("number4 is OK")

#Analize number 5
iter = 0
for object in number5:
    data.append(Element())
    data[i].number = '5'
    data[i].image, data[i].feature = extraction(object)
    i += 1
    iter += 1
print("number5 is OK")

#Analize number 6
iter = 0
for object in number6:
    data.append(Element())
    data[i].number = '6'
    data[i].image, data[i].feature = extraction(object)
    i += 1
    iter += 1
print("number6 is OK")

#Analize number 7
iter = 0
for object in number7:
    data.append(Element())
    data[i].number = '7'
    data[i].image, data[i].feature = extraction(object)
    i += 1
    iter += 1
print("number7 is OK")

#Analize number 8
iter = 0
for object in number8:
    data.append(Element())
    data[i].number = '8'
    data[i].image, data[i].feature = extraction(object)
    i += 1
    iter += 1
print("number8 is OK")

#Analize number 9
iter = 0
for object in number9:
    data.append(Element())
    data[i].number = '9'
    data[i].image, data[i].feature = extraction(object)
    i += 1
    iter += 1
print("number9 is OK")

print("Complete analysis of the Train database")

#KNN
print("\nInitialization KNN")

# Element to analize
#Remember to apply Transformation.py when you want to evaluate a new image.
test = Element()

for aux in range(81):

    name = './Images/image'+str(aux)+'.png'
    image = io.imread(name)
    
    ##COUNTING OBJECTS WITHIN THE IMAGE WITH CANNY ALGORITHM
    borders = cv2.Canny(image, 10, 140)                                                
    ctns, _ = cv2.findContours(borders, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)     #OpenCV4
    contours = len(ctns)
    
    if (contours != 0): #If it is different from an empty box -> in empty boxes the algorithm marks zero because it does not find anything
        test.image, test.feature = extraction(image)
        test.number = '1' # label initial 

        i = 0
        sum = 0
        for ft in data[0].feature:
            sum = sum + np.power(np.abs(test.feature[i] - ft), 2)
            i += 1
        d = np.sqrt(sum)

        for element in data:
            sum = 0
            i = 0
            for ft in (element.feature):
                sum = sum + np.power(np.abs((test.feature[i]) - ft), 2)
                i += 1    
            element.distance = np.sqrt(sum)
            if (sum < d):
                d = sum
                test.number = element.number
    else:
        test.number = '.'
        
    if (aux == 0): vector =  str(test.number)
    else: vector = vector + str(test.number)
        
print(vector)

#Save in a string all the boxes in the sudoku board 
archivo = open("vector.txt","w")
archivo.write(vector)
archivo.close()
