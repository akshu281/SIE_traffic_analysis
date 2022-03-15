from json.tool import main
import cv2
import numpy as np
import os
from natsort import natsorted 

cap = ''
whT = 320
confThreshold = 0.3
nmsThreshold = 0.3

classesFile = 'coco.names'
classNames = []
with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)
#print(len(classNames))

modelConfiguration = 'yolov3.cfg'
modelWeights = 'yolov3.weights'


net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]* wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs,confThreshold,nmsThreshold)
    count = 0

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,255),2)

        if classIds[i]==1 or classIds[i]==2 or classIds[i]==3 or classIds[i]== 7 or classIds[i]==5:
            count = count + 1 
        

    print('Count..', count)
    cv2.putText(img,'count:'+str(count),
                    (100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(255,0,255),2)

    return count

# while True:
if __name__ == "__main__":

    filesnames = os.listdir('images/')
    files = natsorted(filesnames)
    print(files)
    
    myfile = open('count_new.txt', 'w')

    for i in range(0,len(files)):
        img = cv2.imread('images/'+files[i])
        blob = cv2.dnn.blobFromImage(img, 1/255,(whT,whT),[0,0,0],crop=False)
        net.setInput(blob)
        layerNames = net.getLayerNames()
        #print(layerNames)
        outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
        #print(outputNames)
        #print(net.getUnconnectedOutLayers())
        outputs = net.forward(outputNames)
        count = findObjects(outputs,img)

        myfile.write("%s\n" % count)

        # cv2.imwrite('Image.jpg', img)
    # cv2.waitKey(1)

    myfile.close()