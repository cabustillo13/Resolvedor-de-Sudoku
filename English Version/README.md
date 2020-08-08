# Resolvedor-de-Sudoku

**I posted this code in:** https://medium.com/@carlosbustillo/solve-sudoku-9x9-with-computer-vision-and-a-constraint-satisfaction-algorithm-in-python-7bb27769c1eb

Program in python 2.7 to solve a sudoku puzzle from the Android app "Sudoku" from genina.com.
A screenshot of the game is taken (a 720x1280 image is obtained), then useful information is extracted from artificial vision and later it is analyzed with a restriction satisfaction algorithm with backtracking.

**Libraries**

* cv2
* numpy
* matplotlib
* skimage
* mahotas

## **How does it work?**
**1) Run Preprocessing.py**

Extract each sudoku square individually and save them sequentially as photo # .png (where # goes from 0 to 80).
Images of 80x75 pixels are obtained.

```python Preprocessing.py```

**2) Run Transformation.py**

Cut out the borders of each box, in case there is any black border that can be inferred in our analysis.
56x51 pixel images are obtained

```python Transformation.py```

**3) Run Main.py**

Analyze what number is in the box.
In this case, Canny algorithm is used to determine if there is a number or it is an empty box.
Then through the KNN algorithm it is determined which number is in the box.
For the extraction of characteristics, the moments of Hu: 1 and 2, Gaussian filter for filtering and unsupervised thresholding were used.

```python Main.py```

**4) Run Solver.py**

Solve the sudoku game.
A restriction satisfaction algorithm with backtracking is presented.
For this stage, take this repository as a reference: https://github.com/jorditorresBCN/Sudoku

```python Solver.py```

**5) Run Interface.py**

Improves the way the solution is displayed compared to the original screenshot.

```python Interface.py```

## **Run entire program in bash**

Run the play.sh script on your console, to run the entire program in one step if you want.
It depends on the console you have, the way it is run.
In my console I run it as:

```sh ./play.sh```

## **Other programs:**

**6) Functions.py**

It contains all the functions that are used for image preprocessing and transformation.

**7) vector.txt**

It contains all the elements extracted from the screenshot (where the boxes were crossed from left to right, from top to bottom).
The KNN algorithm presented a performance of 97% with respect to all the images analyzed in the Test. In case of an error in the recognition of the numbers, there is the option of manually changing a prediction of the box in the vector.txt.

**8) image.txt**

Contains the path of the image to be analyzed.

**9) Solution.npy**

It is a dictionary that contains the Sudoku solution.


    

    
      

      

    
    



  







  



  

      
        

        
        

  
    
  
