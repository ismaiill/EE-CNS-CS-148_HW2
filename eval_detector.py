import os
import json
import numpy as np
import random
from matplotlib import pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''
    S= ( (box_1[3]-box_1[1])*((box_1[2]-box_1[0])) ) + ( (box_2[3]-box_2[1])*((box_2[2]-box_2[0])))
    iou=0
    if box_2[0] <= box_1[2] and box_1[1] <= box_2[1] <= box_1[3]:
        
        if box_2[3] >= box_1[3] and box_2[2] <= box_1[2]:  #case 1-a-I
                interarea=(box_1[3]-box_2[1] ) * (box_2[2]-box_2[0])
                iou =  interarea/ (S - interarea)  
                
        if box_2[3] >= box_1[3] and box_2[2] >= box_1[2]:  #case 1-b-II
                interarea=(box_1[3]-box_2[1])*(box_1[2]-box_2[0])
                iou =  interarea/ ( S- interarea ) 
                
                
        if box_2[2] >= box_1[2] and box_2[3] <= box_1[3]:   #case 1-b-III
                interarea= (box_2[3]-box_2[1])*(box_1[2]-box_2[0])
                iou =  interarea/ (S- interarea) 
        if  box_2[3] <= box_1[3] and   box_2[2] <= box_1[2]: #cases 1-b-IV
                interarea = (box_2[3]-box_2[1])*((box_2[2]-box_2[0]))  # box_2 inside box_1
                iou =  interarea/ (S- interarea) 
                
    if  box_2[1] >= box_1[3]: #case 2 (outside)
        interarea=0
        iou =  0 
    if  box_2[0] >= box_1[2]:  
        interarea=0
        iou =  0 
    if  box_2[2] <= box_1[0]:
        interarea=0
        iou =  0 
    if  box_2[3]<= box_1[1]:   
        interarea=0
        iou =  0
    if box_2[0] <= box_1[0] and box_1[1] <= box_2[1] <= box_1[3]: 
        
        if box_2[3] <= box_1[3] and box_2[2] <= box_1[0]:  
                #interarea=0
                iou =  0
                
        if box_2[3] >= box_1[3] and box_2[2] <= box_1[0]:  
                #interarea=
                iou =  0
                
                
        if box_2[3] <= box_1[3] and box_2[2] >= box_1[0]: 
                interarea= (box_2[3]-box_2[1])*(box_2[2]-box_1[0])
                iou =  interarea/ (S- interarea) 
                
        if  box_2[3] >= box_1[3] and box_2[2] >= box_1[0]: 
                interarea = (box_1[3]-box_2[1])*((box_2[2]-box_1[0]))  
                iou =  interarea/ (S- interarea)   
                
    # box_2 on the north-west, west or wouth-west of box_1; use BR point to               
        
    if  box_1[0]<= box_2[2] <= box_1[2] and box_1[1] <= box_2[3] <=  box_1[3]:
        if box_2[0] >= box_1[0] and box_2[1]<= box_1[1]:
             interarea = (box_2[2]-box_2[0])*(box_2[3]-box_1[1])
             iou =  interarea/ (S- interarea) 
             
        if box_2[0] <= box_1[0] and box_2[1]<= box_1[1]:
             interarea = (box_2[3]-box_1[1])*(box_2[2]-box_1[0])
             iou =  interarea/ (S- interarea) 
             
    if box_2[2] >= box_1[2] and box_1[1] <= box_2[3] <=  box_1[3]:          
        if box_2[0] <= box_1[2] and box_2[1] <= box_1[1]:
            interarea = (box_2[3]-box_1[1])*(box_1[2]-box_2[0])
            iou =  interarea/ (S- interarea)
    
    assert (iou >= 0) and (iou <= 1.0)
    
    return iou




def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    '''
    BEGIN YOUR CODE
    '''
    for pred_file, pred in preds.items():
        gt = gts[pred_file]
        for i in range(len(gt)):
            for j in range(len(pred)):
                iou = compute_iou(pred[j][:4], gt[i])
                if pred[j][4] >= conf_thr and iou >= iou_thr:
                    TP=TP+1 
                if pred[j][4] >= conf_thr and iou < iou_thr:
                    FP=FP+1
                if  pred[j][4] < conf_thr and iou >= iou_thr:
                    FN=FN+1
                        
    '''
    END YOUR CODE
    '''

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '/Users/Ismail/Documents/Github/Cvision/HW2/hw02_annotations'

# load splits:
split_path = '/Users/Ismail/Documents/Github/Cvision/HW2/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)
    

if done_tweaking:
    
    '''
    Load test data.
    '''
    
    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


#For a fixed IoU threshold, vary the confidence thresholds.
#The code below gives an example on the training set for one IoU threshold. 



# produces an increasing sequnce of confidence thresholds for the plots.

confidence_thrs = [random.randint(1,40) for i in range(40)]
for i in range(40):
    confidence_thrs[i]= confidence_thrs[i] /40
    confidence_thrs=np.sort (confidence_thrs)


# Precision vs Recall for IoU = 0.5 on the training set

tp_train5 = np.zeros(len(confidence_thrs))
fp_train5 = np.zeros(len(confidence_thrs))
fn_train5 = np.zeros(len(confidence_thrs))

Recall5=[]
Precision5=[]

for i in range (len(confidence_thrs)):
    conf1= confidence_thrs[i]
    #print(conf)
    tp_train5[i], fp_train5[i], fn_train5[i] = compute_counts(preds_train, gts_train, iou_thr=0.4, conf_thr=conf1)
    Recall5.append( tp_train5[i] / (tp_train5[i]+fn_train5[i] ) )
    Precision5.append ( tp_train5[i]/ (tp_train5[i]+ fp_train5[i]) )
    
    Precisonarrary5 = np.array(Precision5)
    Recallarray5= np.array(Recall5)
    
    Precisonarrary5= np.delete (Precisonarrary5,0)
    Recallarray5 = np.delete (Recallarray5,0)

PR5 = plt.plot(Recallarray5, Precisonarrary5 )

#plt.show()




# Precision vs Recall for IoU = 0.25 on the training set

tp_train25 = np.zeros(len(confidence_thrs))
fp_train25 = np.zeros(len(confidence_thrs))
fn_train25 = np.zeros(len(confidence_thrs))

Recall25=[]
Precision25=[]

for i in range (len(confidence_thrs)):
    conf1= confidence_thrs[i]
    #print(conf)
    tp_train25[i], fp_train25[i], fn_train25[i] = compute_counts(preds_train, gts_train, iou_thr=0.26, conf_thr=conf1)
    Recall25.append( tp_train25[i] / (tp_train25[i]+fn_train25[i] ) )
    Precision25.append ( tp_train25[i]/ (tp_train25[i]+ fp_train25[i]) )
    
    Precisonarrary25 = np.array(Precision25)
    Recallarray25= np.array(Recall25)
    

PR25 = plt.plot(Recallarray25, Precisonarrary25)



# Precision vs Recall for IoU = 0.65 on the training set

tp_train25 = np.zeros(len(confidence_thrs))
fp_train25 = np.zeros(len(confidence_thrs))
fn_train25 = np.zeros(len(confidence_thrs))

Recall25=[]
Precision25=[]

for i in range (len(confidence_thrs)):
    conf1= confidence_thrs[i]
    #print(conf)
    tp_train25[i], fp_train25[i], fn_train25[i] = compute_counts(preds_train, gts_train, iou_thr=0.65, conf_thr=conf1)
    Recall25.append( tp_train25[i] / (tp_train25[i]+fn_train25[i] ) ) # compute Recall
    Precision25.append ( tp_train25[i]/ (tp_train25[i]+ fp_train25[i]) ) # compute Precision
    
    Precisonarrary25 = np.array(Precision25)
    Recallarray25= np.array(Recall25)
    

PR65 = plt.plot(Recallarray25, Precisonarrary25)




#Plot training set PR curves

plt.legend(["iou_thr=0.5", "iou_thr=0.25","iou_thr=0.75"])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()


if done_tweaking:

########## we repeat exactly the same code used for the training set ########  
        
# produces an increasing seqeunce of confidence thresholds for the plots.
    
    confidence_thrs = [random.randint(1,40) for i in range(40)]
    for i in range(40):
        confidence_thrs[i]= confidence_thrs[i] /40
        confidence_thrs=np.sort (confidence_thrs)


# Precision vs Recall for IoU = 0.5 on the training set

    tp_test5 = np.zeros(len(confidence_thrs))
    fp_test5 = np.zeros(len(confidence_thrs))
    fn_test5 = np.zeros(len(confidence_thrs))

    Recall5=[]
    Precision5=[]

    for i in range (len(confidence_thrs)):
        conf1= confidence_thrs[i]
        #print(conf)
        tp_test5[i], fp_test5[i], fn_test5[i] = compute_counts(preds_test, gts_test, iou_thr=0.5, conf_thr=conf1)
        Recall5.append( tp_train5[i] / (tp_train5[i]+fn_train5[i] ) )
        Precision5.append ( tp_train5[i]/ (tp_train5[i]+ fp_train5[i]) )
        
        Precisonarrary5 = np.array(Precision5)
        Recallarray5= np.array(Recall5)
        
    print( Recallarray5)
    print (Precisonarrary5 )
    
    PR5 = plt.plot(Recallarray5, Precisonarrary5 )

    plt.show()
    
    
    
    # Plot training set PR curves
    
    plt.legend(["iou_thr=0.5", "iou_thr=0.25","iou_thr=0.75"])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()
