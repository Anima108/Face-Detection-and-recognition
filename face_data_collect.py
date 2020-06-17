#Write a python script that captures images from your webcam video stream
#Extracts all faces from the image frame (using haarcascades)
#store the info into numpy arrays

#1. Read and show vid stream,capture images
#2.Detect faces and show bounding box (haarcascade)
#3.Flatten the largest face image (gray scale-to save memory) and save in a numpy array if multiple faces
#4.Repeat the above for multiple people to create multiple data
import cv2
import numpy as np

#Init camera
cap=cv2.VideoCapture(0)
#Face detection
face_cascade= cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0
face_data= []
dataset_path= './data/'
file_name= input ("Enter the name of the person: ")
while True:
	
	ret,frame= cap.read()

	if ret == False:
		continue #try again

	gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	

	faces=face_cascade.detectMultiScale(frame,1.3,5) #list of tuples
	#print(faces)
	faces=sorted(faces,key=lambda f:f[2]*f[3],reverse=True) # to sort the faces on the basis of w*h i.e area; w=f[2],h=f[3]
														    # the largest face being the first in the sorted list

	#or not sort in reverse and start sorting by faces[-1:] start from largest face i.e last ele in the list
	for face in faces:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		#Extract (Crop out the required face) : Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100)) #third dimension in shape-3 channels rgb
		

		cv2.imshow("Face Section",face_section)

		skip += 1
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data))


	cv2.imshow("Frame",frame)
	

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break
		
#Convert our face list array into a numpy array
face_data= np.asarray(face_data) #python list to numpy array KNN 
face_data=face_data.reshape((face_data.shape[0],-1)) #reshape into 2-D Array because KNN 
print(face_data.shape) #(faces,30000) 30000 due to rgb 10000 if grayscale

#Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data) #new file created in data folder
print("Data successfully saved at"+dataset_path+file_name+'.npy')
cap.release()
cv2.destroyAllWindows()


#-offset------------------------
#10 pixels padding of 10 extra pixels around the face so that not only 
				  #we get the face by 10 extra pixels from each point on each side
				  # ---------------
				  #|||||||||||||||  = 10 pixels on each |
				  #----------------

