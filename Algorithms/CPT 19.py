# CPT 19
from random import randint
from allmodules import *
from scipy.ndimage import interpolation, filters
from skimage.filter import threshold_otsu, threshold_adaptive
import scipy
import numpy

def preprocess(image):
    npimg = getImageAsNumpy(image)
    npimg = filters.median_filter(npimg, 3)
    npimg = greyThreshold(npimg, 120)
    return npimg

def segment(npimg, parts=5):
    #return snakeSegmentation(npimg)
    
    [a,b] = [0,0]
    size = npimg.shape
    for c in range(size[1]):
        if 0 in npimg[:,c]:
            a = c
            break
    for c in reversed(range(size[1])):
        if 0 in npimg[:,c]:
            b = c
            break
    print "End points found:", a,b

    avg_size = (b-a)/parts
    prev = a
    charCuts = [a]
    for i in range(parts):
        if prev+avg_size <=b:
            charCuts.append(prev+avg_size)
            prev = prev+avg_size
        else:
            break

    images = []
    for i in range(len(charCuts)-1):
        images.append(npimg[:,charCuts[i]:charCuts[i+1]])
    print "Images Segmented:: ", len(images)
    return images
 
def recognize(images):
    '''
    takes the characters list and returns
    top results from the neural network
    '''

    chars = "" # result string
    net = loadFromPickle("C:\Python27\Captcha_Cracker\Networks\CPT 19.txt")
    ds = loadFromPickle("C:\Python27\Captcha_Cracker\Datasets\CPT 19.txt")
    for im in images:
        act_vect = createInputVector(im,30,25) #get activation vector
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
        #saveIM(images, "CPT", "C:\Python27\Captcha_Cracker\Segmented_Characters\CPT 19\\")
        
        r=""
        original_images.append(getImageAsNumpy(image))
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

#run("C:\Python27\Captcha_Cracker\Captcha Images\CPT 19", 51)
