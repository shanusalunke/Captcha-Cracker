# CPT 6

from allmodules import *
from scipy.ndimage import interpolation, filters
from skimage.filter import threshold_otsu, threshold_adaptive

def preprocess(image):
    
    npimg = getImageAsNumpy(image)
    size = npimg.shape
    
    #npimg = filters.median_filter(npimg, 7)
    npimg = filters.gaussian_filter(npimg, 1.5)
    
    degree = -23
    npimg = 255-npimg
    npimg = interpolation.rotate(npimg, degree, reshape=False) #angle in degrees
    npimg = 255-npimg

    for i in range(size[0]):
        for j in range(size[1]):
            e = npimg[i,j]
            if e < 120 or e > 140:
                npimg[i,j] = 0
            else:
                npimg[i,j] = 255

    #npimg = erode(npimg, 1)
    npimg = invertBackground(npimg)
    npimg = dilate(npimg, 3)
    #npimg = filters.median_filter(npimg, 2)
    
    #npimg = filters.median_filter(npimg, 3)
    #npimg = greyThreshold(npimg, 60)
    #nping = dilate(npimg, 3)
    
    #hist, d = numpy.histogram(npimg, 255)

    #plt.imshow(npimg, cmap=cm.Greys_r)
    #plt.show()
    
    return npimg

def segmentByParts(npimg, parts):
    size = npimg.shape

    for c in range(size[1]):
        if 255 in npimg[:,c] and 0 not in npimg[:,c]:
            continue
        else:
            startcol = c
            break

    for c in reversed(range(size[1])):
        if 255 in npimg[:,c] and 0 not in npimg[:,c]:
            continue
        else:
            endcol = c
            break
    print "start and end col", startcol, endcol
    avg_char = (endcol-startcol)/parts #average size of character
    projection = []
    for c in range(size[1]):
        projection.append(numpy.count_nonzero(npimg[:,c]))

    images = []
    for i in range(1,parts+1):
        print "Images now", len(images)
        axispoint = startcol + (i*avg_char)
        print axispoint
        sub_projection = projection[axispoint-10:axispoint+10]
        print "SUB PROJECTION", sub_projection
        sub_index = sub_projection.index(min(sub_projection))
        if sub_index+startcol < axispoint:
            index = axispoint - sub_index
        else:
            index = axispoint + sub_index
        #index = (projection[axispoint-10:]).index(max_val)
        if startcol >= 0 and index <= endcol and startcol!=index:
            images.append(npimg[:,startcol:index])
        startcol = index
    print "segmented", len(images), "images"
    print images[0]
    return images

def segment(npimg):
    #return segmentByParts(npimg, 4)
    images = snakeSegmentation(npimg, 10)
    print "Returned from Snake", len(images), "images"
    #images = singleColumnSegmentation(npimg, 5)

    # Trying to split large images
    average=0
    for i in range(len(images)):
        temp = images[i]
        size = temp.shape
        average = average + size[1]
    average = average / len(images)

    finalim = []
    
    for index,img in enumerate(images):
        s = img.shape
        im_size = s[1]
        #print im_size, average+10, im_size > (average+10)
        if im_size > (average+ 15):
            projection = []
            print "TOO BIG... SEGMENTATION IS SPLITTING IT", index
            for i in range(im_size):
                projection.append(numpy.count_nonzero(img))
            print "im_size", im_size
            a = (im_size/2) - 10
            b = (im_size/2) + 10
            temp_p = projection[a:b]
            split = (im_size/2)+ temp_p.index(min(temp_p))
            #im1 = img[:,0:s[1]/2]
            #im2 = img[:,s[1]/2:s[1]]
            print "length", len(projection), " Split:", split
            im1 = img[:,0:split]
            im2 = img[:,split:s[1]]

            #plt.subplot(211)
            #plt.imshow(im1, cmap=cm.Greys_r)
            #plt.subplot(212)
            #plt.imshow(im2, cmap=cm.Greys_r)
            #plt.show()
            
            #images[index]=im1
            #images.insert(index+1,im2)
            finalim.append(im1)
            finalim.append(im2)
        else:
            finalim.append(img)
    
    if len(finalim) <=2:
        print "call recursively"
        npimg = greyThreshold(npimg, 100)
        npimg = erode(npimg, 1)
        #plt.imshow(npimg, cmap=cm.Greys_r)
        #plt.show()
        finalim = segment(npimg)

    return finalim

def recognize(images):
    '''
    takes the characters list and returns
    top results from the neural network
    '''

    chars = "" # result string
    net = loadFromPickle("C:\Python27\Captcha_Cracker\Networks\CPT 6.txt")
    ds = loadFromPickle("C:\Python27\Captcha_Cracker\Datasets\CPT 6.txt")
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

#run("C:\Python27\Captcha_Cracker\Captcha Images\CPT 6", 1)
