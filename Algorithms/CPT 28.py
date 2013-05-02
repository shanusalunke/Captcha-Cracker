# CPT 28

from allmodules import *
from scipy.ndimage import interpolation, filters
from skimage.filter import threshold_otsu, threshold_adaptive


def run(path, number):
        
    path = "C:\Python27\Captcha Cracker\Captcha Images\CPT 28"

    imageFiles = getRawImages(path)
    ch = 'n'

    for image in imageFiles:
        
        npimg = getImageAsNumpy(image)
        plt.subplot(221)
        plt.imshow(npimg, cmap=plt.cm.gray)
        
        npimg = greyThreshold(npimg, 127) 
        npimg = invertBackground(npimg)
        npimg = erode(npimg, 1)
        npimg = filters.median_filter(npimg, 3)
        
        plt.subplot(222)
        plt.imshow(npimg, cmap=plt.cm.gray)
        plt.show()
        
        ch = input("next?")
        if ch != 'y':
            break


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
        
