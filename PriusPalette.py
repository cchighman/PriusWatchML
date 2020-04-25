import numpy as np

class PriusPalette:
    def __init__(self):

        self.requiredShades = []
        

        #8db1b7 <-> #50757f -- rgb(50,87,77) <-> rgb(181,227,223)
        self.dullBlueGrey = [([77,87,50], [223,217,181])]
        #+++Required Palettes: 353  Average Color: 17  PCA Colors: 70  Perfect Match: 4  Total: 30
        self.requiredShades.append(self.dullBlueGrey) 

        #d3e8eb <-> #84bac8 -- rgb(133,186,200) <-> rgb(222,232,235)
        self.lightBlueShades = [([200,186,133], [235,232,222])]
        #+++Required Palettes: 333  Average Color: 0  PCA Colors: 57  Perfect Match: 0  Total: 25/31
        #self.requiredShades.append(self.lightBlueShades)
                         
        #rgb(111,142,193)                        
        #51707b - rgb(81,112,123) -- rgb(71,102,113) <-> rgb(91,122,143)
        self.shade4 = [([113,102,71], [193,142,111])]
        #   -> Required Palettes: 25  Average Color: 16  PCA Colors: 0  Perfect Match: 0  Total: 30
        self.requiredShades.append(self.shade4)      
        
        #5a6676 - rgb(90,102,118) -- rgb(90,102,118) <-> rgb(150,172,178)
        self.shade6 = [([118,102,90], [178,172,150])]
        # +++Required Palettes: 329  Average Color: 10  PCA Colors: 0  Perfect Match: 0  Total: 30/31
        self.requiredShades.append(self.shade6)
   
        #345359 - rgb(52,83,89) -- rgb(32,63,69) <-> rgb(82,103,109)
        self.blue_green_shade = [([69,63,32], [109,103,72])]
        # +++Required Palettes: 354  Average Color: 26  PCA Colors: 68  Perfect Match: 14  Total: 30/31 / 29/33
        self.requiredShades.append(self.blue_green_shade)
        
        #649ca0 - rgb(100,156,160) -- rgb(80,136,140) <-> rgb(120,176,180)
        self.shade12 = [([140,136,80], [180,176,120])]
        # +++Required Palettes: 81  Average Color: 0  PCA Colors: 1  Perfect Match: 0  Total: 29
        self.requiredShades.append(self.shade12)
   
        ########
   
        #badbd5 - rgb(186,219,213) -- rgb(176,209,203) <-> rgb(196,229,223)
        self.shade5 = [([203,209,176], [223,229,196])]
        # +++Required Palettes: 245  Average Color: 0  PCA Colors: 0  Perfect Match: 0  Total: 27/31
        self.requiredShades.append(self.shade5)
        #d1f7f7 - rgb(209,247,247) -- rgb(199,237,237) <-> rgb(219,255,255)
        self.shade14 = [([237,237,199], [255,255,219])]
        # +++Required Palettes: 118  Average Color: 0  PCA Colors: 0  Perfect Match: 0  Total: 27
                     
        #7b99b6 - rgb(123,153,182) -- rgb(113,143,172) <-> rgb(133,163,192)
        self.shade7 = [([172,143,113], [192,163,133])]
        # +++Required Palettes: 114  Average Color: 0  PCA Colors: 0  Perfect Match: 0  Total: 30/31
        self.requiredShades.append(self.shade7)
        
        #c7e3ec - rgb(199,227,236) -- rgb(189,217,226) <-> rgb(209,237,246)
        self.shade8 = [([226,217,189], [246,237,209])]
        # +++Required Palettes: 198  Average Color: 0  PCA Colors: 0  Perfect Match: 0  Total: 27/31
                
        #9baeb7 - rgb(155,174,183) -- rgb(145,164,173) <-> rgb(165,184,193)
        self.shade9 = [([173,164,145], [193,184,165])]
        # +++Required Palettes: 286  Average Color: 0  PCA Colors: 0  Perfect Match: 0  Total: 30/31
        self.requiredShades.append(self.shade9)
                
    def required_shades(self):
        return self.requiredShades

    def has_shade(self,shade):
        outputList = []   
        for boundary in self.requiredShades:                    
            for (lower, upper) in boundary:
                lower = np.array(lower, dtype = "uint8")
                upper = np.array(upper, dtype = "uint8")
                aboveLower = shade[0] >= lower[0] and shade[1] >= lower[1] and shade[2] >= lower[2]
                aboveUpper = shade[0] <= upper[0] and shade[1] <= upper[1] and shade[2] <= upper[2]
                #print("Shade: " + str(shade) + "  Palette Range: " + str(boundary) + "  Match? " + str(aboveLower and aboveUpper))
                outputList.append(aboveLower and aboveUpper)
                
        return any(outputList)
