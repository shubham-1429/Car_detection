
import cv2

#carcascade_src = 'cars.xml'
#video_src = 'dataset/video1.avi' 
video_src = 'dataset/video2.avi' #copy dataset folder path

cap = cv2.VideoCapture(video_src) #if we are using camera then use 0
car_cascade = cv2.CascadeClassifier('cars.xml')

while True:
    #ret,img = cap.read() #ret and img are variables t
    _,img = cap.read() #if u dont want variable then use "_",both are correct 
    if (type(img) == type(None)):
        break
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)      
    
    cv2.imshow('video', img)
    
    if cv2.waitKey(33) == 27:
        break

cv2.destroyAllWindows()