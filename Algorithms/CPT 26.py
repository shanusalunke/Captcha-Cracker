# CPT 26
from random import randint
from allmodules import *
from scipy.ndimage import interpolation, filters
from skimage.filter import threshold_otsu, threshold_adaptive
import scipy
import numpy

def preprocess(image):
    npimg = getImageAsNumpy(image)
    #npimg = filters.gaussian_filter(npimg, 1)

    size =npimg.shape
    for r in range(size[0]):
        for c in range(size[1]):
            if npimg[r,c] > 235 and c>0:
                npimg[r,c] = npimg[r,c-1]
    
    npimg = greyThreshold(npimg, 130)
    #npimg = erode(npimg, 1)
    #npimg = skeleton(npimg)
    #npimg = filters.median_filter(npimg, 3)
    #npimg = filters.gaussian_filter(npimg, 1)
    
    return npimg

def segment(npimg_im, denominator=7):
    npimg = numpy.copy(npimg_im)
    size = npimg.shape
    projection = []
    temp = numpy.copy(npimg)
    for c in range(size[1]):
        projection.append(size[1]-numpy.count_nonzero(npimg[:,c]))
    threshold = 209

    for index,p in enumerate(projection):
        if p<=threshold:
            npimg[:,index]=120
    #plt.imshow(npimg,cmap=cm.Greys_r)
    #plt.show()
    
    breakpoints = []
    images = []
    
    # iterating over last row to find breakpoints
    for i in range(size[1]):
        if npimg[size[0]-2,i] == 120:
            breakpoints.append(i)

    if (i-1) not in breakpoints:
        breakpoints.append(i-1) #appending the last column
    # splitting image into components
    charthreshold = size[1]/denominator
    lastcol = 0
    for i in breakpoints:
        diffcol = i - lastcol
        if diffcol > charthreshold:
            images.append(temp[:,lastcol:i])
            lastcol = i

    #removing small, noisy segments
    finalim=[]
    for im in images:
        wt = float(numpy.count_nonzero(im))
        tot = float(im.shape[0] * im.shape[1])
        bl_percent = ((tot-wt)/tot)*100
        if bl_percent >10:
            finalim.append(im)
        else:
            print "Segmentation is removing noise"
    #plt.imshow(npimg, cmap=cm.Greys_r)
    #plt.show()
    return finalim
    '''
    plt.subplot(211)
    plt.imshow(npimg,cmap=cm.Greys_r)
    plt.subplot(212)
    plt.plot(projection)
    plt.show()
    ''' 
            
    '''
    images = snakeSegmentation(npimg,7)
    print "Snake Segmentation split it into", len(images), "images"
    '''
    #return images
 
def recognize(images):
    '''
    takes the characters list and returns
    top results from the neural network
    '''

    chars = "" # result string
    net = loadFromPickle("C:\Python27\Captcha_Cracker\Networks\CPT 26.txt")
    ds = loadFromPickle("C:\Python27\Captcha_Cracker\Datasets\CPT 26.txt")
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
        npimg = preprocess(imageFiles[i])
        images = segment(npimg)
        answer = recognize(images)

        r=""
        original_images.append(getImageAsNumpy(imageFiles[i]))
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

#run("C:\Python27\Captcha_Cracker\Captcha Images\CPT 26",1000)
