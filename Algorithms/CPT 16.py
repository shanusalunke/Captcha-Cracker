# CPT 16

from allmodules import *
from scipy.ndimage import interpolation, filters
from skimage.filter import threshold_otsu, threshold_adaptive

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
    
def run(path, number):
    path = "C:\Python27\Captcha Cracker\Captcha Images\CPT 16"

    imageFiles = getRawImages(path)
    ch = 'n'

    for image in imageFiles:
        
        npimg = getImageAsNumpy(image)

        npimg = greyThreshold(npimg, 127)
        npimg2 = npimg
        plt.subplot(211)
        plt.imshow(npimg2, cmap=plt.cm.gray)
        
        #find row number at colour transition, store in list
        #find column numbers at col transition, store in list (col 0 and last col are always there)
        #for every [0,col+1], if the col is black, invert the slice npimg[0:r1,cols[i]:cols[i+1]-1]

        size = npimg.shape
        h_split = []
        v_split = [0]

        curr = npimg[0,0]
        for r in range(1,size[0]):
            n = npimg[r,0]
            if n != curr:
                h_split.append(r)
                curr = n

        curr = npimg[size[0]-1,0]
        for c in range(1,size[1]):
            n = npimg[size[0]-1,c]
            if n!=curr:
                curr = n
                #npimg[:,c] = 127
                v_split.append(c-1)

        '''
        #splitting across center
        r = h_split[0]
        m1 = [[255,0],[0,255]]
        m2 = [[0,255],[255,0]]
        for c in range(1,size[1]):
            n = npimg[r-1:r+1,c-1:c+1]
            if (n==m1).all() or (n==m2).all():
                npimg[:,c] = 127
                v_split.append(c)

        '''
        print h_split, v_split, size
        '''
        for i in range(2):
            for j in range(len(v_split) -1):
                if(i==0):
                    r2 = h_split[0] #i.e. 24
                    r1 = 0
                else:
                    r1 = h_split[0]+1
                    r2 = size[0]-1
                c = v_split[j]
                col = npimg[r2,c]
                print "REGION", i+j, ":: Colour", col , "AT:: [", r1,":",r2,",",v_split[j],":",v_split[j+1],"]"
                if col == 0:
                    npimg[r1:r2, v_split[j]:v_split[j+1]] = 255-npimg[r1:r2, v_split[j]:v_split[j+1]]

        '''
        for i in range(2):
            for j in range(len(v_split)-1):
                if(i==0):
                    r2 = h_split[0]-1 #i.e. 24
                    r1 = 0
                else:
                    r1 = h_split[0]
                    r2 = size[0]-1
                print "REGION", i+j, "AT:: [", r1,":",r2,",",v_split[j],":",v_split[j+1],"]"
                npimg[r1:r2,v_split[j]:v_split[j+1]] = invertBackground(npimg[r1:r2,v_split[j]:v_split[j+1]])
                              
        plt.subplot(212)
        plt.imshow(npimg, cmap=plt.cm.gray)
        plt.show()
        
        ch = input("next?")
        if ch != 'y':
            break
        
