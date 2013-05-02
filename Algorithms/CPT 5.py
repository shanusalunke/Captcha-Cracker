# CPT 5

from allmodules import *
import scipy
from scipy.ndimage import interpolation, filters
from skimage.filter import threshold_otsu, threshold_adaptive
import numpy

def preprocess(image):
    npimg = getImageAsNumpy(image)
    npimg = greyThreshold(npimg, 127)
    npimg = invertBackground(npimg)
    npimg = erode(npimg, 1)
    #npimg = skeleton(npimg)
    return npimg
        
def segment(image):
    '''
    images = closeSegmentation(npimg)
    for i in images:
        plt.imshow(i)
        plt.show()
    '''
    size = image.shape
    height_thresh = size[1]/15
    image = 255-image
    mask = image > 1
    #mask = removeSmall(mask, mask.mean())
    #mask = 255-mask
    labels, nb = ndimage.label(mask)
    #print labels
    #labels = removeSmall(labels, labels.mean())
    segmented = []
   
    for i in range(1,nb):
        slice_x, slice_y = ndimage.find_objects(labels==i)[0]
        print slice_x, slice_y
        roi = image[slice_x, slice_y]
        remove_pixel = removeSmall(roi, roi.mean())
        s = remove_pixel.shape
        print s, height_thresh
        if s[1] >= height_thresh:
            segmented.append(remove_pixel)
        #segmented.append(roi)
    print segmented
    #plt.imshow(labels, cmap=cm.jet)
    #plt.show()
    return segmented
    
    
def saveIM(images, name):
    
    for index, ims in enumerate(images):
        nname = 'C:\Python27\Captcha_Cracker\Segmented_Characters\CPT 5\\'+name+ str(index)+".png"
        scipy.misc.imsave(nname, ims)
        print "saved"  


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
'''
print "done importing"
arg_list = (sys.argv)
path = arg_list[1]
number = arg_list[2]
print "path", path
print "number", number
run(path, int(number))
'''
run("C:\Python27\Captcha Cracker\Captcha Images\CPT 5", 1)
