import cv2
import numpy as np
import sqlite3
import urllib.request as url
import webbrowser

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
#cam=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("recognizer\\trainingData.yml")
id=0
def getProfile(id):
    conn=sqlite3.connect("FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile
def webopen(a,b,c,d):
    if(a=="Deepak"):
        url2="file:///C:/Users/Asus/Desktop/FDproject/faceRecognitionSqlite/USer.html"
        webbrowser.open(url2, autoraise=True)
        cv2.destroyAllWindows()
    elif(a=="Deepak mandal"):
        url3="file:///C:/Users/Asus/Desktop/User2.html"
        webbrowser.open(url3,autoraise=True)
        cv2.destroyAllWindows()


font=cv2.FONT_HERSHEY_SIMPLEX
while(True):
    #ret,img=cam.read();
    url2="file:///C:/Users/Asus/Desktop/New%20Text%20Document.html"
    cam=url.urlopen("http://192.168.43.1:8080/shot.jpg")
    imgNp=np.array(bytearray(cam.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)
    img=cv2.resize(img,(640,480))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=5,minSize=(100,100),flags=cv2.CASCADE_SCALE_IMAGE);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(conf<65):
            profile=getProfile(id);
            if(profile!=None):
                cv2.putText(img,str(conf),(x,y+h+0),font,1.0,(255,0,0),lineType=cv2.LINE_AA);
                cv2.putText(img,str(profile[1]),(x,y+h+30),font,1.0,(255,0,0),lineType=cv2.LINE_AA);
                cv2.putText(img,str(profile[2]),(x,y+h+60),font,1.0,(255,0,0),lineType=cv2.LINE_AA);
                cv2.putText(img,str(profile[3]),(x,y+h+90),font,1.0,(255,0,0),lineType=cv2.LINE_AA);
                cv2.putText(img,str(profile[4]),(x,y+h+120),font,1.0,(255,0,0),lineType=cv2.LINE_AA);
                name=str(profile[1])
                age=str(profile[2])
                gender=str(profile[3])
                cr=str(profile[4])
                webopen(name,age,gender,cr);
        else:
            cv2.putText(img,str(conf),(x,y+h+0),font,1.0,(255,0,0),lineType=cv2.LINE_AA);
            cv2.putText(img,str("Unknown"),(x,y+h+30),font,1.0,(255,0,0),lineType=cv2.LINE_AA);
    cv2.imshow("Face",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
