#Recognise face using some classification algo - like logistic,KNN,SVM, etc.
#1.load the training data (numpy arrays of all the persons)
        #x- vals are stored in np arrays
        #y- vals we need to assign for each person
#2.Read a vid stream using opencv
#3.extract faces out of it (Testing)
#4.use knn to find the prediction of face(int)
#5.map the predicted id to name of the user
#6.Display the predictions on the screen - bounding box and name

#Using KNN
import cv2
import numpy as np
import os

##KNN#######################################################
def distance(v1,v2):
    return np.sqrt(sum((v1-v2)**2)) #basically the distance formula

def knn(train,test,k=5):#k=5 no of neighbours in consideration
    dist=[]
    
    
    for i in range(train.shape[0]):
        # Get the vector and label
        ix= train[i, :-1] #features
        iy= train[i,-1] #last column for labels
        
        #Compute the dist from test point
        d= distance(test,ix)
        dist.append([d,iy])
    #Sort based on dist and get top k
    dk= sorted(dist, key=lambda x:x[0])[:k]
    #Retrive only the labels 
    labels= np.array(dk)[:,-1]
    
    #Get freq of each label

    output=np.unique(labels,return_counts=True)
    #Find max freq of and corr label
    index= np.argmax(output[1])
    return output[0][index]
    print(type(d))
##############################################################

#Init camera
cap=cv2.VideoCapture(0)
#Face detection
face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip=0
dataset_path= './data/'
face_data= [] #training, x

labels= [] #y

class_id=0 #labels
names= {} #mapping b/w id and name

#Data prep

for fx in os.listdir(dataset_path): #listdir what all files? in data folder
	if fx.endswith('.npy'): #(x,30000)
		names[class_id]=fx[:-4]#mapping b/w class id and name    #slice .npy
		print("loaded"+fx)
		data_item=np.load(dataset_path+fx)
		face_data.append(data_item) # first,second so on faces for a given file

		#Create Labels for the Class
		target= class_id*np.ones((data_item.shape[0]))
		class_id+=1
		labels.append(target)

face_dataset= np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape(-1,1)
print(face_dataset.shape)
print(face_labels.shape)

trainset= np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape) # 30000 features +1 labels=30001



#Testing


while True:
	ret,frame=cap.read()

	if ret == False:
		continue


	faces=face_cascade.detectMultiScale(frame,1.3,5)


	for face in faces:
		x,y,w,h= face

		#get region of interest
		offset=10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		out= knn(trainset,face_section.flatten()) #or reshape

		#Display on the screen the name and rec
		pred_name=names[int(out)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA) #adds text to img,x,y.. is coord,font,size?,color,thick
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)



	cv2.imshow("faces",frame)

	key=cv2.waitKey(1) &0xFF
	if key==ord('q'):
		break

cap.release()
cap.destroyAllWindows()

