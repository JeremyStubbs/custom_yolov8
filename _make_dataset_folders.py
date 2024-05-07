import csv
import os
import numpy as np
from PIL import Image 


"""
Create folder structure needed for trainings
"""

def get_train_and_test_lists ():

    with open('labels.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    # Make list of 75 lists
    csv_info = []
    for value in range(0,102): 
        csv_info.append([])

    # print(csv_info)

    # Fill it
    for line in data:
        csv_info[int(line[1])-1].append(int(line[0]))

    # The data is now organized in an array that contains 102 arrays
    # Each individual array in order contain the picture numbers for that label i.e. csv_info[0] has images of label #1
    # Except my labels start at 0 because it's easier to work with and those brits didn't publish a list of labels to flower names so it doesn't matter
    # print("Indices of all images with label = 1 ", csv_info[0])
    csv_info = csv_info[:75]
    print(len(csv_info))

    train_list = []

    for value in range(0,75): 
        train_list.append([])

    # Convert numbers into filepaths
    for value in range(0,75):
        for pic in csv_info[value]:
            train_list[value].append('image_'+ str(pic).zfill(5)+'.jpg')

    base_path = '.\\images\\jpg\\'
    dest_path = '.\\dataset\\custom_data\\'

    os.mkdir(dest_path)

    for value in range(0,75): 
        os.mkdir(dest_path+str(value))
        for img in train_list[value]:
            temp = Image.open(base_path+img)
            temp.save(dest_path+str(value)+'\\'+img, "JPEG")

    # print( train_list)

get_train_and_test_lists()