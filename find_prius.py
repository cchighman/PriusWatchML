import numpy as np
import cv2
import os

def image_has_shade(boundaries):
    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output.any() > 0

def has_required_shades():
    shadeList = []
    for shade in requiredShades:
        shadeList.append(image_has_shade(shade))
    return all(shadeList)

boundaries = [
    # lower B, G, R  < upper B, G, R 
    #([178,160,10], [200,185,110]),
    ([240,220,10], [255,255,150]),
    #([209,195,140], [229,215,162]),
    #([189,164,87], [209,184,107]),
    #([126,109,35], [146,129,55])
]

dullBlueGrey = [
    #8db1b7 <-> 50757f
    ([127,117,80], [183,177,141])    
]

darkBlueShades = [
    #56a3be <-> 84bac8
    ([190,163,86], [200,186,133])    
]

lightBlueShades = [
    #d3e8eb <-> 84bac8
    ([200,186,133], [235,232,222])    
]


requiredShades = []
requiredShades.append(dullBlueGrey)
requiredShades.append(darkBlueShades)
requiredShades.append(lightBlueShades)

path = "./frames2/"

arr = os.listdir(path)
for image in arr:
    print("Checking " + image)
    file = path + image
    image = cv2.imread(file)

    if has_required_shades() is True:
        print("Image " + file + " has required shades.\n")
        
