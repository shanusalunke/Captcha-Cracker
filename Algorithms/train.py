from allmodules import *
'''
def createDS(vectors):
    # overriding allmodules.createDS(vect, no)
    ni = len(vectors[0][0])
    no = len(vectors[0][1])  

    labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9']
    #creating classifiction dataset
    ds = ClassificationDataSet(ni,nb_classes=no, class_labels=labels)
    return ds
'''
def run(cpt_name, set_type):
    if set_type <0 or set_type >6:
        print "Invalid set type"
        return 0
    
    vect = loadVectors("C:\Python27\Captcha_Cracker\Dataset Vectors\\" + cpt_name+".txt")
    net = myNetwork(vect)

    # Trying dataset creation
    ds = createDS(vect, set_type)
    ds = setRecordsInDS(ds, vect)

    print "begin training"
    
    #tstdata, trndata = createTrnTst(ds)
    trainer, trained = train(net, ds)
    
    saveAsPickle(net, "C:\Python27\Captcha_Cracker\Networks\\" + cpt_name+".txt")
    saveAsPickle(ds, "C:\Python27\Captcha_Cracker\Datasets\\" + cpt_name+".txt")

#%1- filename %2-cpt_name %3-set_type
arg_list = (sys.argv)
cpt_name = arg_list[1]
set_type = arg_list[2]
print "CPT Name", cpt_name
print "Set Type:", set_type
run(cpt_name, int(set_type))
