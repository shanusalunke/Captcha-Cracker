from allmodules import *
import numpy as np
import sys

'''
    Type 0- lower case
    Type 1- upper case
    Type 2- numbers 
    Type 3- lower + upper
    Type 4-lower + nos
    Type 5- upper + nos
    Type 6- lower+upper+nos

'''
def addVector(path):
    count =0
    folders = os.listdir(path)
    for j in folders:
        if j !="small":
            imageFiles = getRawImages(path+j+"\\")
            for i,image in enumerate(imageFiles):
                count = count+1
                npimg = getImageAsNumpy(image)
                ip = createInputVector(npimg, 30,25) #returns list
                op = createOutputVector(j, int(set_type))
                saveVectors(ip, op, "C:\Python27\Captcha_Cracker\Dataset Vectors\\" + cpt_name + ".txt")
                print "saved"
    print "saved " , count, " objects"
    
def run(path, cpt_name, set_type):
    if set_type <0 or set_type >6:
        print "Invalid set type"
        return 0
    if set_type == 3 or set_type == 6:
        #lower case and upper case together. lower case will be saved in folder named 0
        addVector(path+"small\\")
        addVector(path)
    else:
        addVector(path)
    

#%1- filename %2-path %3-cpt_name %4- set_type
arg_list = (sys.argv)
path = arg_list[1]
cpt_name = arg_list[2]
set_type = arg_list[3]
print "path", path
print "name", cpt_name
print "Set Type:", int(set_type)
run(path, cpt_name, int(set_type))
