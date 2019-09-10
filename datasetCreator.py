import cv2
import numpy as np
import sqlite3
import urllib
import urllib.request as url

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
#cam=cv2.VideoCapture(0);

def insertOrUpdate(id,Name,age,gender,cr):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    isRecordExist=0
    for row in cursor:
        isRecordExist=1
    if(isRecordExist==1):
        cmd="UPDATE People SET Name="+str(Name)+",Age="+str(age)+" ,Gender="+str(gender)+" ,Criminal record="+str(cr)+" WHERE ID="+str(id)
    else:
        cmd="INSERT INTO People Values("+str(id)+","+str(Name)+","+str(age)+","+str(gender)+","+str(cr)+ ")"
    conn.execute(cmd)
    conn.commit()
    conn.close()

id=input('enter user id')
Name=input('enter your name')
age=input("Enter age")
gender=input("Enter gender")
cr=input("Enter any Criminal record")
insertOrUpdate(id,Name,age,gender,cr)

sampleNum=1;
while(True):
    #ret,img=cam.read();
    cam=url.urlopen("http://192.168.43.1:8080/shot.jpg")
    imgNp=np.array(bytearray(cam.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)
    img=cv2.resize(img,(640,480))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(100,100),flags=cv2.CASCADE_SCALE_IMAGE);
    for(x,y,w,h) in faces:
        sampleNum=sampleNum+1;
        cv2.imwrite("dataSet/User."+str(id)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow("Face",img);
    cv2.waitKey(100)
    if(sampleNum>100):
        cam.release()
        cv2.destroyAllWindows()
        break
