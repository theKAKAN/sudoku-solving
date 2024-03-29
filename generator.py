#Loading Libraries

# To make sure it works in headless mode
import matplotlib as mpl
mpl.use("Agg")

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, random
import cv2
from glob import glob
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img
from keras.utils.np_utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from PIL import Image

from helper import solveSudokuPuzzle


# function to greyscale, blur and change the receptive threshold of image
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (3,3),6) 
    #blur = cv2.bilateralFilter(gray,9,75,75)
    threshold_img = cv2.adaptiveThreshold(blur,255,1,1,11,2)
    return threshold_img
def main_outline(contour):
    biggest = np.array([])
    max_area = 0
    for i in contour:
        area = cv2.contourArea(i)
        if area >50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i , 0.02* peri, True)
            if area > max_area and len(approx) ==4:
                biggest = approx
                max_area = area
    return biggest ,max_area
def reframe(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4,1,2),dtype = np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis =1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new
def splitcells(img):
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes
def printPuzzle(quiz):
    result = ""
    for row in range(9):
        if row % 3 == 0 and row != 0:
            result += ".................... <br>"

        for col in range(9):
            if col % 3 == 0 and col != 0:
                result += "| "

            if col == 8:
                result += str(quiz[row][col]) + " <br> "
            else:
                result += str(quiz[row][col]) + " "
    return result

class Solver:
    model = tf.keras.models.load_model('./output/model.keras')
    def __init__(self, image_path):
        self.image_path = image_path
        self.puzzle = cv2.imread(self.image_path)
    
    def preprocessImage(self):
        self.puzzle_nop = cv2.resize(self.puzzle, (450,450))
        # Preprocessing Puzzle 
        self.puzzle = preprocess(self.puzzle_nop)
        plt.figure()
        plt.imshow(self.puzzle_nop)
        plt.savefig('./static/fig/step1.png')
    
    def findContours(self):
        # Finding the outline of the sudoku puzzle in the image
        su_contour_2= self.puzzle.copy()
        su_contour, hierarchy = cv2.findContours(self.puzzle,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        black_img = np.zeros((450,450,3), np.uint8)
        su_biggest, su_maxArea = main_outline(su_contour)
        if su_biggest.size != 0:
            su_biggest = reframe(su_biggest)
            cv2.drawContours(su_contour_2,su_biggest,-1, (0,255,0),10)
            su_pts1 = np.float32(su_biggest)
            su_pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
            su_matrix = cv2.getPerspectiveTransform(su_pts1,su_pts2)
            su_imagewrap = cv2.warpPerspective(self.puzzle_nop,su_matrix,(450,450))
            su_imagewrap =cv2.cvtColor(su_imagewrap, cv2.COLOR_BGR2GRAY)

        plt.figure()
        plt.imshow(su_imagewrap)
        plt.savefig('./static/fig/step2.png')

        self.imageWrap = su_imagewrap
    
    def genCells(self):
        sudoku_cell = splitcells(self.imageWrap)
        def CropCell(cells):
            Cells_croped = []
            for image in cells:
                img = np.array(image)
                img = img[4:46, 6:46]
                img = Image.fromarray(img)
                Cells_croped.append(img)

            return Cells_croped

        self.sudoku_cell = CropCell(sudoku_cell)
        self.sudoku_cell = Solver.readCells(self.sudoku_cell)
        self.sudoku_cell = np.reshape(self.sudoku_cell,(9,9))

        return printPuzzle(self.sudoku_cell)
    
    def solve(self):
        if( solveSudokuPuzzle(self.sudoku_cell, 0, 0) ):
            return printPuzzle(self.sudoku_cell)
        else:
            return "Solution doesn't exist. Model misread digits, probably."
    
    @staticmethod
    def readCells(cell):
        result = []
        for image in cell:
            # preprocess the image as it was in the model 
            img = np.asarray(image)
            img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
            img = cv2.resize(img, (32, 32))
            img = img / 255
            img = img.reshape(1, 32, 32, 1)
            # getting predictions and setting the values if probabilities are above 65% 

            predictions = Solver.model.predict(img)
            classIndex = np.argmax(predictions,axis=1)
            probabilityValue = np.amax(predictions)

            if probabilityValue > 0.65:
                result.append(classIndex[0])
            else:
                result.append(0)
        return result
    
    @staticmethod
    def getModelSummary():
        return Solver.model.summary()