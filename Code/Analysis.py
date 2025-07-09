import numpy as np
import os
import cv2
import datetime
import pickle

now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

martix_pixelIndex = np.zeros((378, 504, 19), dtype=int)
martix_classPixelIndex = np.zeros((19, 378, 504), dtype=int)
martix_calssIndex = np.zeros((19), dtype=np.int64)
martix_calssImageCount = np.zeros((19), dtype=np.int64)
martix_calssImageIdentifier = np.zeros((19,9000), dtype='bool')

labelPath = "./dataset/indexLabel/"
filenames = [f for f in os.listdir(labelPath) if os.path.isfile(os.path.join(labelPath, f))]

count = 0
for filename in filenames:
    count+=1
    print(f"{count}/{len(filenames)}")
    downscale_factor = 4
    image = cv2.imread(os.path.join(labelPath, filename), cv2.IMREAD_GRAYSCALE)
    new_size = (image.shape[1] // downscale_factor, image.shape[0] // downscale_factor)
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
    
    martix_calssAdded = np.zeros((19), dtype=int)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            k = image[i, j]
            martix_pixelIndex[i, j, k] += 1
            martix_classPixelIndex[k, i, j] += 1
            martix_calssIndex[k] += 1
            if martix_calssAdded[k] != 1:
                martix_calssImageIdentifier[k][count] = True
                martix_calssAdded[k] = 1
                martix_calssImageCount[k] += 1

file = open("martix_pixelIndex.pickle", "wb")
pickle.dump(martix_pixelIndex, file)
file.close()
file = open("martix_classPixelIndex.pickle", "wb")
pickle.dump(martix_classPixelIndex, file)
file.close()
file = open("martix_calssIndex.pickle", "wb")
pickle.dump(martix_calssIndex, file)
file.close()
file = open("martix_calssImageCount.pickle", "wb")
pickle.dump(martix_calssImageCount, file)
file.close()
file = open("martix_calssImageIdentifier.pickle", "wb")
pickle.dump(martix_calssImageIdentifier, file)
file.close()

totalPixel = image.shape[1] * image.shape[0] * len(filenames)
with open("Analysis.txt", "w") as text_file:
    text_file.write("Label Analysis\n\n")
    for i in range(19):
        text_file.write(f"Class {i}: {(martix_calssIndex[i] / totalPixel * 100):.2f}% -- {martix_calssIndex[i]}/{totalPixel}\n")
print("Label Analysis")

totalPic = len(filenames)
with open("Analysis_Class.txt", "w") as text_file:
    text_file.write("Class Analysis\n\n")
    for i in range(19):
        text_file.write(f"Class {i}: {(martix_calssImageCount[i] / totalPic * 100):.2f}% -- {martix_calssImageCount[i]}/{totalPic}\n")
print("Class Analysis")

for z in range(19):
    for prob in [0.01 ,0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9]:
        pic = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if martix_classPixelIndex[z][i][j] > len(filenames) * prob:
                    pic[i,j] = 255
                
        cv2.imwrite(f"ClassDistri_C{z}-{prob}.png", cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB))
        print(f"ClassDistri_C{z}-{prob}.png")

for a in range(19):
    pic = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if martix_classPixelIndex[a][i][j] > 0:
                pic[i][j] = 255
    
    contours, _ = cv2.findContours(pic, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        maxArea = -1
        maxAreaIndex = None
        for z in range(len(contours)):
            area = cv2.contourArea(contours[z])
            if area > maxArea:
                maxArea = area
                maxAreaIndex = z

        pic = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        if maxAreaIndex is not None:
            cv2.drawContours(pic, [contours[maxAreaIndex]], -1, 255, thickness=cv2.FILLED)
        
    cv2.imwrite(f"ClassMax_C{a}.png", cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB))
    print(f"ClassMax_C{a}.png")

now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
