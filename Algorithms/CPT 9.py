#CPT 9

'''
from PIL import Image
import ImageFilter
from pylab import *
import numpy as np
from matplotlib import pyplot as plt
import os # for the lstdir function
from scipy import ndimage
'''
from allmodules import *
from scipy.ndimage import interpolation, filters

def preprocess(image):
    npimg = getImageAsNumpy(image)
    npimg = filters.gaussian_filter(npimg, 1)
    #npimg = greyThreshold(npimg, 200)
    #npimg = filters.median_filter(npimg,3)
    #npimg = invertBackground(npimg)
    npimg = erode(npimg, 2)
    npimg = greyThreshold(npimg, 165)
    return npimg

def segment(npimg):
    return singleColumnSegmentation(npimg,10)

def recognize(images):
    '''
    takes the characters list and returns
    top results from the neural network
    '''

    chars = "" # result string
    net = loadFromPickle("C:\Python27\Captcha_Cracker\Networks\CPT 9.txt")
    ds = loadFromPickle("C:\Python27\Captcha_Cracker\Datasets\CPT 9.txt")
    for im in images:
        act_vect = createInputVector(im, 30,25) #get activation vector
        chars = chars + predictChar(act_vect, net, ds, 5) + "    "
        
    return chars

def run(path, number):
    print "Start Running Algorithm"
    #check whether results directory is clean
    checkDirectory()
    
    imageFiles = getRawImages(path)
    i=0
    j=0
    original_images = []
    processed_images = []
    results = []
    result=[]
    
    for image in imageFiles:
        j = j+1 #count to check number of captchas cracked
        i = randint(0,len(imageFiles))
        npimg = preprocess(image)
        images = segment(npimg)
        answer = recognize(images)
        #saveIM(images, "CPT 9", "C:\Python27\Captcha_Cracker\Segmented_Characters\CPT 9\\")
        r=""
        original_images.append(getImageAsNumpy(image)
        processed_images.append(npimg)
        results.append(answer)
        r = r+answer[0]
        for index,c in enumerate(answer):
            if c ==' ' and index<len(answer)-1:
                r = r + answer[index+1]
        result.append(r)
        #plt.imshow(npimg, cmap=cm.Greys_r)
        #plt.title(answer)
        #plt.show()
        if j>=number:
            break
    createResult(original_images, processed_images, results, result)

print "done importing"
arg_list = (sys.argv)
path = arg_list[1]
number = arg_list[2]
print "path", path
print "number", number
run(path, int(number))

#run("C:\Python27\Captcha_Cracker\Captcha Images\CPT 9", 1)



