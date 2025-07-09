import os
import shutil
import numpy as np
import re
import random

def sample_without_replacement(arr, n):
    if n > len(arr):
        raise ValueError("")
    
    result = []
    for _ in range(n):
        index = random.randrange(len(arr))
        result.append(arr.pop(index))
    return result

#spilt in to 20 folders, each one containing 5%
folderNumDict = {
    "K-01":[78] * 16 + [79] * 4,
    "K-03":[195] * 7 + [196] * 13, 
    "V-01":[37] * 17 + [38] * 3, 
    "V-02":[41] * 7 + [42] * 13, 
    "V-03":[92] * 15 + [93] * 5
}
    # "K-01": 1564 -> 78*16 + 79*4
    # "K-03": 3913 -> 195*7 + 196*13
    # "V-01": 743  -> 37*17 + 38*3
    # "V-02": 833  -> 41*7 + 42*13
    # "V-03": 1845 -> 92*15 + 93*5
print("------- File num Check -------")
for key in folderNumDict.keys():
    temp = 0
    for value in folderNumDict[key]:
        temp += value
    print(f"{key}: {temp}")
print("------- File num Check -------")

fileDict = {
    "K-01":[],
    "K-03":[], 
    "V-01":[], 
    "V-02":[], 
    "V-03":[]
}

print("------- File length Check -------")
for folder in folderNumDict.keys():
    indexLabelPath = f"./{folder}/indexLabel/"
    indexLabelFiles = [f for f in os.listdir(indexLabelPath) if os.path.isfile(os.path.join(indexLabelPath, f))]
    fileDict[folder] = indexLabelFiles

    print(f"Folder: {folder} --> {len(fileDict[folder])}")
print("------- File length Check -------")
input("enter to continue")

for datasetNum in range(1,21):
    folderPath = f"./Datasets/5P-{datasetNum:02d}/"
    for folder in folderNumDict.keys():
        indexLabelPath = f"./{folder}/indexLabel/"
        imagePath = f"./{folder}/image/"
        labelPath = f"./{folder}/label/"
        selectedFile = sample_without_replacement(fileDict[folder], folderNumDict[folder][datasetNum-1])
        print(f"select {folderNumDict[folder][datasetNum-1]} for 5P-{datasetNum:02d} from {folder}")
        for file in selectedFile:
            srcPath = indexLabelPath + file
            dstPath = folderPath + "indexLabel/" + file
            shutil.move(srcPath, dstPath)
            #print(f"Copy [indexLabel]: {srcPath} ---> {dstPath}")

            srcPath = imagePath + file
            dstPath = folderPath + "image/" + file
            shutil.move(srcPath, dstPath)
            #print(f"Copy [image]: {srcPath} ---> {dstPath}")

            srcPath = labelPath + file
            dstPath = folderPath + "label/" + file
            shutil.move(srcPath, dstPath)
            #print(f"Copy [label]: {srcPath} ---> {dstPath}")

print("------- End Check -------")
print(fileDict)
for folder in folderNumDict.keys():
    print(f"Folder: {folder} --> {len(fileDict[folder])}")
print("------- End Check -------")