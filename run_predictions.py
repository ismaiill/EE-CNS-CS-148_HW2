import os
import numpy as np
import json
from PIL import Image

def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)

    '''
    BEGIN YOUR CODE
    '''
    N=20
    (T_n_rows,T_n_cols,K_n_channels) = np.shape(T)
    heatmap = np.random.random((20, 20))
    for i in range(19):
        for j in range(19):
            if np.shape(I)==(480,640,3):
                subimage = I[int(480/N)*i:int(480/N)*i+T_n_rows,int(640/N)*j:int(640/N)*j+T_n_cols]
                v_1 = np.matrix(subimage.ravel())
                v_2 = np.matrix(T.ravel())
                normalizedv_1 = v_1/np.linalg.norm(v_1)
                normalizedv_2 = v_2/np.linalg.norm(v_2)
                score = np.inner(normalizedv_1, normalizedv_2)
                heatmap[i,j]=score
                
    
    '''
    END YOUR CODE
    '''

    return heatmap

def predict_boxes(heatmap):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    #box_height = 8
    #box_width = 6

    #num_boxes = np.random.randint(1,5)

    #for i in range(num_boxes):
        #(n_rows,n_cols,n_channels) = np.shape(I)

        #tl_row = np.random.randint(n_rows - box_height)
        #tl_col = np.random.randint(n_cols - box_width)
        #br_row = tl_row + box_height
        #br_col = tl_col + box_width

        #score = np.random.random()

        #output.append([tl_row,tl_col,br_row,br_col, score])    
    N=20
    T_n_rows=24
    T_n_cols=32
    for i in range(19):
        for j in range(19): 
            output.append([int(480/N)*i,int(640/N)*j,int(480/N)*i+T_n_rows,int(640/N)*j+T_n_cols,heatmap[i,j]])
            

    '''
    END YOUR CODE
    '''

    return output


def detect_red_light_mf(I):
    
    output=[]
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    #template_height = 8
    #template_width = 6

    # You may use multiple stages and combine the results
    N=20
    T = Image.open("kernel.jpg")
    T=T.resize((int(640/N),int(480/N)))
    T = np.asarray(T)
    T=T[:,:,:3]

    heatmap = compute_convolution(I, T)
    
    for i in range(len(  predict_boxes(heatmap)   )):
        if predict_boxes(heatmap)[i][4] > 0.5:   # 0.5 is the treshhold to detect the bounding box
            output.append(predict_boxes(heatmap)[i])
    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output



# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '/Users/Ismail/Documents/Github/Cvision/HW1/RedLights2011_Medium'

# load splits: 
split_path = '/Users/Ismail/Documents/Github/Cvision/HW2/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    preds_train[file_names_train[i]] = detect_red_light_mf(I)
    
print(I[1].shape)    

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
