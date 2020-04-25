import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])


boundaries = [
    # lower B, G, R  < upper B, G, R 
    #([178,160,10], [200,185,110]),
    #([240,220,10], [255,255,150]),
    #([209,195,140], [229,215,162]),
    #([189,164,87], [209,184,107]),
    #([126,109,35], [146,129,55])  
    ([127,117,80], [183,177,141]),
    
]

dullBlueGrey = [
    #8db1b7 <-> 50757f
    ([127,117,80], [183,177,141])    
]

darkBlueShades = [
    #56a3be <-> 84bac8
    ([190,163,86], [200,186,133])    
]



for (lower, upper) in boundaries:
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    
mask = cv2.inRange(image, lower, upper)


output = cv2.bitwise_and(image, image, mask = mask)

if(output.any() > 0):
    print("Prius Detected.")

cv2.imshow("images", np.hstack([image, output]))
cv2.waitKey(0)