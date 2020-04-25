#
#  Find the average color in an image
#
import os
import cv2
import numpy as np
import time
import datetime
import argparse

from PriusImage import PriusImage
from PriusPalette import PriusPalette


ap = argparse.ArgumentParser()
ap.add_argument("-pp", "--predicted-path", required=False)
ap.add_argument("-sp", "--sample-path", required=False)
args = vars(ap.parse_args())



def compare_images():
    predictedPath = args["predicted_path"]
    samplePath = args["sample_path"]

    predictedImages = os.listdir(predictedPath)
    sampleImages = os.listdir(samplePath)

    
    shadePredictedCount = 0
    avgPredictedCount = 0
    pcaPredictedCount = 0
    totalPredictedCount = 0

    shadeSampleCount = 0
    avgSampleCount = 0
    pcaSampleCount = 0
    totalSamleCount = 0

    
    print("\nPrius Matches\n")
    for img in predictedImages:          
            
        start = time.time()
        try:
            print("\nColor Profile for " + img)
            path = args["predicted_path"] + img
            
            priusImage = PriusImage(path)
            palette = PriusPalette()
            
            totalPredictedCount = totalPredictedCount + 1
            if priusImage.has_required_shades():
                shadePredictedCount = shadePredictedCount + 1

            avg = priusImage.avg_color()
            avgInPalette = palette.has_shade(avg)
            print("Average Color: " + str(avg) + "  Palette Match? " + str(avgInPalette))
            if avgInPalette:
                avgPredictedCount = avgPredictedCount + 1
            
            requiredColors = []
            for color in priusImage.pca_colors():
                requiredColors.append(palette.has_shade(color))
            print("PCA Analysis\n")
            if any(requiredColors):
                pcaPredictedCount = pcaPredictedCount + 1                
                #print("Successfully validated palette, average, and PCA for " + img)
        except Exception as e:print(e)
        end = time.time()
        print("\nRequired Palettes: " + str(shadePredictedCount) + "  Average Color: " + str(avgPredictedCount) + "  PCA Colors: " + str(pcaPredictedCount) + "  Total: " + str(totalPredictedCount)+ "  Time: " + str(end-start) + "\n")

    
    if args["sample_path"] is not None:
        print("\nPrius Matches\n")
        for img in SampleImages:          
                
            start = time.time()
            try:
                print("\nColor Profile for " + img)
                path = args["sample-path"] + img
                
            priusImage = prius_image.PriusImage(path)
            palette = prius_palette.PriusPalette()
                
            totalSampleCount = totalSampleCount + 1
            if priusImage.has_required_shades():
                shadeSampleCount = shadeSampleCount + 1
                avg = priusImage.avg_color()
                avgInPalette = palette.has_shade(avg)
                print("Average Color: " + str(avg) + "  Palette Match? " + str(avgInPalette))
                if avgInPalette:
                    avgSampleCount = avgSampleCount + 1
                
                requiredColors = []
                for color in priusImage.pca_colors():
                    requiredColors.append(palette.has_shade(color))
                print("PCA Analysis\n")
                if any(requiredColors):
                    pcaSampleCount = pcaSampleCount + 1                
                    #print("Successfully validated palette, average, and PCA for " + img)
            except Exception as e:print(e)
            end = time.time()
            print("\nRequired Palettes: " + str(shadeSampleCount) + "  Average Color: " + str(avgSampleCount) + "  PCA Colors: " + str(pcaSampleCount) + "  Total: " + str(totalSampleCount)+ "  Time: " + str(end-start) + "\n")

compare_images()             
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    