# Importing Necessary Libraries
import numpy as np
import cv2
import json
import os
import time
import sys
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.models import load_model

# Defining Predictor Class
class Classifier:
    def __init__(self,base_model,class_model,class_list):
        if base_model=="VGG16":
            self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=(160,120,3))
        elif base_model=="ResNet":
            self.base_model =  ResNet50(weights='imagenet', include_top=False, input_shape=(160,120,3))
        else:
            print("Error: Base Model Not Defied")
        self.class_model=load_model(class_model)
        self.class_list=class_list
        
    def feature_extraction(self,epoch_time):
        vid_name="Tester"
        
        for i in range(0,10):
            vid=vid_name
            cap = cv2.VideoCapture(vid+'.mp4')
            if(cap.isOpened()):
                break
            epoch_time+=1
        
#        Checking Empty Video
        
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if(frameCount==0):
            return epoch_time,-1
        
        X = np.empty((frameCount, 160, 120, 3), np.dtype('uint8'))
        fc = 0
        ret = True
        
#        Frame Extraction Initiated
        
        while (fc < frameCount  and ret):
            ret, frame = cap.read()
            frame=cv2.resize(frame,(120,160))
            X[fc]=frame
            fc += 1   
#            cv2.imshow("Frame",frame)
#            cv2.waitKey(40)
        cap.release()
        cv2.destroyAllWindows()
        
#        Frame Extraction Complete, Starting Prediction 
        
        x_features = self.base_model.predict(X)
        x_features = x_features.reshape(x_features.shape[0],
                     x_features.shape[1] * x_features.shape[2]*
                     x_features.shape[3])
        return x_features
    
    def classify(self,x_features):
        pred = self.class_model.predict(x_features)
        return pred

if __name__=="__main__":
    while True:
        epoch_time=sys.argv[0]
        Model=Classifier("ResNet",'ResNet_4_(25).h5',['Anomaly','Normal'])
        x_features=Model.feature_extraction(0)
        if(type(x_features)==int):
            break
        np.save('X_Train0',x_features[:,0:7680])
    
    # Converting Prediction Data to suitable format 
    l=len(pred)
    num_partitions=2
    Arr1=[]
    for i in range(num_partitions):
        Arr1.append(np.average(pred[i*int(l/num_partitions):(i+1)*int(l/num_partitions),0]))
    data={"Anomaly":Arr1,"Time_Taken":[int((end-start)/num_partitions)]*num_partitions}
    
    # Saving Data in File
    File_loc="Data"
    with open(File_loc+".json","w") as f:
        json.dump(data,f)
    epoch_time+=500