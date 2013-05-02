#REMEMBER:
    # Include an invert clause in threholding. If the background is w-on-b instead of b-on-w, invert colours
    # when using '' => \\'
    # when using "" => \"   
    # keep mynet, myds, etc as global variables or private class var ?
    # add init_module funcs that import all required libs (eg. init_learning: import pybrain)

#from skimage.morphology import watershed, is_local_maximum
from skimage.morphology import *
from PIL import Image
import ImageFilter
from pylab import *
import numpy as np
from matplotlib import pyplot as plt
import os # for the lstdir function
from scipy import ndimage
import scipy
import numpy
import shutil

import pickle

from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where, misc
from numpy.random import multivariate_normal
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

def invertBackground(npimg):
    '''
    Takes a numpy array as input
    If the image is NOT black on white (as is standard)
    The image is inverted

    ASSUMPTION: The background is larger than the foregoround
    '''
    size = npimg.shape
    imarea = size[0]*size[1]
    whitearea = count_nonzero(npimg)
    blackarea = imarea - whitearea
    if whitearea < blackarea:
        #implies the pic is white on black and needs
        # to be inverted
        npimg = 255-npimg
    return npimg

def removeSmall(im, size):
    '''
    Remove small regions from a numpy image
    inut image is black on white (std)
    inversion is done internally
    Image for labeling needs to be White on black
    The image returned is inverted (black on white )
    '''
    #im = 255-im
    label_im, nb_labels = ndimage.label(im)
    mask = im>im.mean()
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mean_vals = ndimage.sum(im, label_im, range(1, nb_labels + 1))
    mask_size = sizes < size
    remove_pixel = mask_size[label_im]
    return remove_pixel
 
def getRawImages(mypath):
    '''
    Creates a list of all jpg and png images in a specified folder
    Returns the list

    The list contains the PATHS and not images
    '''
    import os
    imageFiles = []
    imageFiles += [mypath+'\\'+each for each in os.listdir(mypath) if (each.endswith('.png')or each.endswith('.gif') or each.endswith('jpg')or each.endswith('bmp'))]
    return imageFiles

def greyThreshold(npimg, threshold):
    '''
    Takes a numpy array of the image and threshold as input and returns
    the binarized numpy array as output
    '''
    size = npimg.shape
    for row in range(size[0]):
        for col in range(size[1]):
            if npimg[row,col] <= threshold:
                npimg[row,col] = 0
            else:
                npimg[row,col] = 255
    
    return npimg

def getImage(impath):
    '''
    Takes the path of  an image and opens it.
    Also converts the image into 8-bit greyscale

    Requires PIL.Image
    '''
    from PIL import Image
    im = Image.open(impath)
    im = im.convert('L')
    return im

def getImageAsNumpy(impath):
    ''' takes a path gets the PIL image
    converts to numpy array and returns'''
    im = getImage(impath)
    im = array(im)
    return im

def dilate(npimg, elem_rad=1):
    '''
    Takes a numpy image and the structure element's radius.
    Structure Element is assumed to be a disk

    dilates the numpy image and returns
    '''
    elem = disk(elem_rad)
    d = erosion(npimg, elem)
    return d

def erode(npimg, elem_rad=1):
    '''
    Takes a numpy image and the structure element's radius.
    Structure Element is assumed to be a disk

    erodes the numpy image and returns
    '''
    elem = disk(elem_rad)
    d = dilation (npimg, elem)
    return d

def skeleton(npimg):
    '''
    Returns the sketon of the given numpy image.
    Image is black on white (conversion is done here)
    Needs skimage.morphology
    error prone !
    '''
    size = npimg.shape
    for i in range(size[0]):
        for j in range(size[1]):
            if npimg[i,j] > 127:
                npimg[i,j] = 0
            else:
                npimg[i,j] =1
    skel = skeletonize(npimg)
    skel = greyThreshold(skel,0)
    skel = 255-skel
    return skel

def singleColumnSegmentation(npimg, denominator=10, noise_thresh=1):
    print "Using Single Column Segmentation"
    size = npimg.shape
    charthreshold = size[1]/denominator #Any segment of width less than charthreshold is considered noise and ignored
    lastcol = 0
    segpoints =[0] # a list that hods the column numbers corresponding to image beginning and end

    for i in range(size[1]):
        column = npimg[:,i]
        if 255 in column and 0 not in column:
            diffcol = i-lastcol
            if diffcol > charthreshold:
                segpoints.append(i)
                lastcol = i
    
    images=[] #list containing segmented images
    imstart =0
    imend = 0
    black = True
    for i in range(len(segpoints)-1):
        images.append(npimg[:,segpoints[i]:segpoints[i+1]])

    #removing small, noisy segments
    finalim=[]
    for im in images:
        wt = float(numpy.count_nonzero(im))
        tot = float(im.shape[0] * im.shape[1])
        bl_percent = ((tot-wt)/tot)*100
        if bl_percent > noise_thresh:
            finalim.append(im)
            
    print "Images Segmented:", len(finalim)
    return finalim

def singleColumnSegmentation2(npimg, denominator=10):
    '''
    Performs single column segmentation for single white columns
    and returns a list of segmented images
    '''
    
    size = npimg.shape

    charthreshold = size[1]/denominator #Any segment of width less than charthreshold is considered noise and ignored
    lastcol = 0
    segpoints =[0] # a list that hods the column numbers corresponding to image beginning and end


    #print size, charthreshold

    for i in range(size[1]):
        column = npimg[:,i]
        if 255 in column and 0 not in column:
            diffcol = i-lastcol
            '''
            if diffcol >1 and diffcol < charthreshold:
                for j in range(diffcol):
                    npimg[:,lastcol+j] = 0
            elif diffcol > 1:
                # extract image uptil the next complete column
                segpoints.append(i)
            lastcol = i
            npimg[:,i] = 0
            '''
            if diffcol > charthreshold:
                segpoints.append(i)
                lastcol = i
    
    images=[] #list containing segmented images
    imstart =0
    imend = 0
    black = True
    for i in range(len(segpoints)-1):
        images.append(npimg[:,segpoints[i]:segpoints[i+1]])
    '''
    for i in range(size[1]):
        column = npimg[:,i]
        if 0 in column and 255 not in column:
            # sngle black line
            if black== False:
                imend = i
                im = npimg[:,imstart:imend]
                images.append(im)
            black = True
        else:
            #character may start
            if black == True:
                imstart = i
            black = False
    '''
    average=0
    for i in range(len(images)):
        temp = images[i]
        size = temp.shape
        average = average + size[1]
    average = average / len(images)

    for index,img in enumerate(images):
        s = img.shape
        im_size = s[1]
        #print im_size, average+10, im_size > (average+10)
        if im_size > (average+ 10):
            #print "TOO BIG... SEGMENTATION IS SPLITTING IT", index
            im1 = img[:,0:s[1]/2]
            im2 = img[:,s[1]/2:s[1]]
            images[index]=im1
            images.insert(index+1,im2)
    
    return images

def singleRowSegmentation(npimg):
    '''
    Takes a numpy image as input and boxes the character only, removing
    empty columns above and below it
    '''
    print "Using Single Row Segmentation"
    size = npimg.shape
    t=0
    b=size[0]-1
    for r in range(size[0]):
        row = npimg[r,:]
        if 0 not in row:
            t=r
        else:
            break
            
    for r in reversed(range(size[0]-1)):
        row = npimg[r,:]
        if 0 not in row:
            b=r
        else:
            break

    return npimg[t:b,:]

def closeSegmentation(image):
    '''
    uses label method to segment closely spaced characters
    Internally removes small regions
    Assumes npimg is standard (black on white)
    returns list of character images
    '''
    print "Using Close Segmentation"
    image = 255-image
    mask = image > 1
    labels, nb = ndimage.label(mask)
    segmented = []
    for i in range(1,nb):
        slice_x, slice_y = ndimage.find_objects(labels==i)[0]
        roi = image[slice_x, slice_y]
        remove_pixel = removeSmall(roi, roi.mean())
        segmented.append(remove_pixel)

    return segmented

def snakeSegmentation(npimg, denominator=10):
    print "Using Snake Segmentation"
    temp = numpy.copy(npimg)
    size = npimg.shape
    for col in range(size[1]):
        c = col
        for r in range(size[0]):
            if (r+1) < size[0] and (c+1)<size[1]:
                if npimg[r+1,c]== 255:
                    npimg[r,c] = 120
                    continue
                else:
                    if npimg[r+1, c+1]!= 0:
                        npimg[r+1,c+1] = 120
                        c = c+1
                        #r = r-1
                    elif npimg[r+1, c-1] != 0:
                        npimg[r+1,c-1] = 120
                        c = c-1
                    elif npimg[r,c+1] != 0:
                        c = c+1
                        r = r-1
                        npimg[r,c+1] = 120
                    else:
                        break
                        #print "no way"

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
        if bl_percent >1:
            finalim.append(im)
        else:
            print "Snake is removing noise"
    #plt.imshow(npimg, cmap=cm.Greys_r)
    #plt.show()
    return finalim

#def correctSlant()

def padImages(finalim):
    '''
    Pads white rows and columns around each image in the finalim list
    '''
    for npimg in finalim:
        npimg = singleRowSegmentation(npimg)
        for i in range(10):
            r= [255]*(npimg.shape)[1]
            npimg = vstack((npimg,r))
            npimg = vstack((r,npimg))
            c = []
            for cc in range((npimg.shape)[0]):
                c.append([255])
            npimg = hstack((npimg,c))
    return finalim

def padImage(finalim):
    '''
    Pads white rows and columns around ONE image
    '''
    npimg = singleRowSegmentation(finalim)
    for i in range(10):
        r= [255]*(npimg.shape)[1]
        npimg = vstack((npimg,r))
        npimg = vstack((r,npimg))
        c = []
        for cc in range((npimg.shape)[0]):
            c.append([255])
        npimg = hstack((npimg,c))
    return finalim

def createInputVector(npimg, nrow=25, ncol=20):
    '''
    Takes a numpy array as input and creates input vector for the NN
    The npimg will be resized to a standard size (20x25)

    Imports scipy.misc.imresize
    '''
    #im = padImage(npimg)
    im = misc.imresize(npimg, [nrow,ncol])
    im = greyThreshold(im, 127)

    size = im.shape
    invect = [] # the input vector
    for row in range(size[0]):
        for col in range(size[1]):
            if im[row,col] == 0:
                invect.append(0)
            else:
                invect.append(1)
    return invect

def createInputVector2(npimg, nrow=25, ncol=20):
    '''
    ZERNIKE TRIAL:
    Takes a numpy array as input and creates input vector for the NN
    The npimg will be resized to a standard size (20x25)

    Imports scipy.misc.imresize
    '''
    #npimg = singleRowSegmentation(npimg)
    nrow = 25 #number of rows corr to height
    ncol = 20 #number of cols corr to width

    im = misc.imresize(npimg, [nrow,ncol])
    size = im.shape
    invect = [] # the input vector
    for row in range(size[0]):
        for col in range(size[1]):
            if im[row,col] == 0:
                invect.append(0)
            else:
                invect.append(1)
    return invect


def createOutputVector(char, setType):
    '''
    takes a character input and returns the output vector of that
    character

    Type 0- lower case
    Type 1- upper case
    Type 2- numbers 
    Type 3- lower + upper
    Type 4-lower + nos
    Type 5- upper + nos
    Type 6- lower+upper+nos
    
    '''
    #print "SET - TYPE", setType
    if setType == 0:
        labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    elif setType == 1:
        labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    elif setType == 2:
        labels = ['0','1','2','3','4','5','6','7','8','9']
    elif setType == 3:
        labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    elif setType == 4:
        labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
    elif setType == 5:
        labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
    elif setType == 6:
        labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
    else:
        print "An error occured.. no set type defined for the dataset"
        return 0

    opvect = [0]*len(labels)
    index = labels.index(char)
    opvect[index]=1
    return opvect

def saveVectors(ip, op, filename, mode="ab"):
    '''
    creates single vect[] of ip and op vectors and
    saves in filename
    '''
    vect =[]
    vect.append(ip)
    vect.append(op)
    tofile = open(filename,mode)
    pickle.dump(vect,tofile)
    tofile.close()
    
    
def loadVectors(filename):
    '''
    Load all vectors from the given file name
    '''
    fromfile=open(filename,"rb")
    vect = []
    while True:
        try:
            vect.append(pickle.load(fromfile))
        except EOFError:
            #End of file is reached
            break
    fromfile.close()
    return vect

def myNetwork(vectors):
    '''
    Takes the vectors (from loadvectors) as input and creates the NN
    Returns reference of NN
    '''
    #no of ip, op and hidden are calculated
    ni = len(vectors[0][0])
    no =  len(vectors[0][1])
    nh = (ni+no)*2/3
    
    #using the build network shortcut
    net = buildNetwork(ni, nh, no, outclass=SoftmaxLayer)
    print "inputs, outputs, etc", ni, nh, no
    return net

def loadDS(mypath):
    '''
    Load custom dataset from path
    '''

def createDS(vectors, setType):
    '''
    Creates a dataset of a particular type
    Type 0- lower case only
    Type 1- upper case only
    Type 2- numbers only
    Type 3- lower + upper
    Type 4-lower + nos
    Type 5- upper + nos
    Type 6- lower+upper+nos
    '''
    #no of ip, op and hidden are calculated
    ni = len(vectors[0][0])
    no = len(vectors[0][1])
    #nh = (ni+no)*2/3
    #nh = (ni+no)/2

    #choosing class labels
    if setType == 0:
        labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    elif setType == 1:
        labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    elif setType == 2:
        labels = ['0','1','2','3','4','5','6','7','8','9']
    elif setType == 3:
        labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    elif setType == 4:
        labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
    elif setType == 5:
        labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
    elif setType == 6:
        labels = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
    else:
        print "An error occured.. no set type defined for the dataset"
        return 0
    
    #creating classifiction dataset
    ds = ClassificationDataSet(ni,nb_classes=no, class_labels=labels)
    
    return ds

def setRecordsInDS(ds, vect):

    ##############################################
    # Target Dimension of DS is supposed to be 1 #
    ##############################################
    
    for i in range(len(vect)):
        ip = vect[i][0]
        out = vect[i][1]
        op=[]
        #to get the index of the maximum
        high = max(out)
        op.append(out.index(high))
        
        #op.append(out.argmax(axis=0))
        ds.appendLinked(ip,op)
        
    ds._convertToOneOfMany(bounds=[0, 1])
    return ds
'''
def setRecordsInDS(ds, vect):
    for i in range(len(vect)):
        ip = vect[i][0]
        out = vect[i][1]
        op=[]
        op.append(out.argmax(axis=0))
        ds.appendLinked(ip,op)
    ds._convertToOneOfMany(bounds=[0, 1])
    return ds
'''    
def saveAsPickle(item, mypath):
    '''
    Saves given item (be it net, ds or vector) using pickle
    myname and mypath are strings (including the \ and extension)
    Existing files will be overwritten
    '''
    fileobj = open(mypath,'w')
    pickle.dump(item,fileobj)
    fileobj.close()
        
def loadFromPickle(mypath):
    '''
    Load dataset/net/vectors from file at mypath
    '''
    fileobj = open(mypath,'r')
    item = pickle.load(fileobj)
    fileobj.close()
    return item

def train(mynet, ds):
    '''
    Uses backprop trainer to train the net for one epoch
    '''
    trainer = BackpropTrainer( mynet, ds, learningrate=0.01)
    print "Training ... ... ... "

    for i in range(10):
        trained = trainer.train()
        print trained
    print "Training Complete"
    return trainer, trained
'''
def predictChar(invect, mynet, ds, n=5 ):
    
    Takes an input vector and returns the topN characters in
    a list
    
    prediction = mynet.activate(invect)
    #print prediction
    characters = ""
    charcount = n
    #To print the top 5 answers
    sortedans = sorted(prediction)
    l = len(prediction)
    for i in reversed(range(l-n,l)):
        index = nonzero(prediction==sortedans[i])
        characters = characters + (ds.getClass(index[0][0]))
    return characters
'''

def predictChar(invect, mynet, ds, n=5 ):
    '''
    For the sake of this example
    '''
    print "Now predicting characters"
    prediction = mynet.activate(invect)
    characters = ""
    charcount = n
    #To print the top 5 answers
    sortedans = sorted(prediction)
    l = len(prediction)
    for i in reversed(range(l-n,l)):
        index = nonzero(prediction==sortedans[i])
        characters = characters + (ds.getClass(index[0][0]))
    return characters

def createTrnTst(ds, propotion=0.25):
    tstdata, trndata = ds.splitWithProportion( propotion )
    trndata._convertToOneOfMany( )
    tstdata._convertToOneOfMany( )
    return tstdata, trndata

def saveIM(images, name, path):
    for index, ims in enumerate(images):
        nname = path+name+ str(index)+ str(randint(0,500))+".jpg"
        pilim = Image.fromarray(ims)
        pilim.save(nname,"JPEG")
        #scipy.misc.imsave(nname, ims)
        print "saved"  

def saveSingleImage(image, fullPath):
    pilim = Image.fromarray(image)
    pilim.save(fullPath,"JPEG")
    print "saved"

def checkDirectory():
    '''
    Cleans the temp folder in the Results Directory
    '''
    directory = "C:\\Python27\\Captcha_Cracker\\Results\\Temp"
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        #if path exists, delete it and create it again
        shutil.rmtree(directory)
        os.makedirs(directory)
        
def createResult(original_images, processed_images, results, result):
    '''
    Creates the innerHTML that needs to be appended to the mainframe
    in Result.hta
    '''
    #Root directory
    directory = "C:\\Python27\\Captcha_Cracker\\Results\\Temp"
    data = "<form name='result_form' class='result_form'><table class='result_table'>"
    data = data+"<tr><td>Original Image</td><td>Processed Image</td><td>Top-N Results</td><td>Result</td><td>Top-N Check</td><td>Top Check</td></tr>"
    #Save the original and processed images one by one
    for i in range(len(results)):
        orig_name = directory + "\\O"+str(i)+".jpg"
        saveSingleImage(original_images[i],orig_name)
        proc_name = directory + "\\P"+str(i) + ".jpg"
        saveSingleImage(processed_images[i], proc_name)
        data = data +"<tr>"
        data = data + "<td><img class='result_img' src='"+ orig_name +"'></td>"
        data = data + "<td><img class='result_img' src='"+ proc_name +"'></td>"
        data = data + "<td>"+results[i] + "</td><td>" + result[i] +"</td>"
        data = data + "<td> <input type='checkbox' name='topn'></td>"
        data = data + "<td> <input type='checkbox' name='top'></td></tr>"
        
    data = data+"</table></form>"
    #Create the data portion and save as txt
    f = open(directory +"\\temp.txt" , "w")
    f.write(data)
    f.close()
