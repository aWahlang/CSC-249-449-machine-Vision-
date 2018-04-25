import csv
import cv2
import re
import os.path

with open('validation.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    total = 427298
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    count = 1
    for row in readCSV:
        filename = row[0][row[0].index('/')+1:]
        emotion = row[6]
        src = "/home/daniel/Documents/affectnet-manual/" + filename
        
        
        print(str(count) + "/" + str(total) + "%")
        count += 1
        
        if os.path.exists(src) and src.endswith(".jpg"):
            image = cv2.imread(src)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                face = image[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224)) 
        
                if emotion == "0":
                    dst = "outputs/val/neutral/" + filename
                    cv2.imwrite(dst, face)
                    
                elif emotion == "1":
                    dst = "outputs/val/happy/" + filename
                    cv2.imwrite(dst, face)
                    
                elif emotion == "2":
                    dst = "outputs/val/sad/" + filename
                    cv2.imwrite(dst, face)
                    
                elif emotion == "3":
                    dst = "outputs/val/surprise/" + filename
                    cv2.imwrite(dst, face)
                    
                elif emotion == "4":
                    dst = "outputs/val/fear/" + filename
                    cv2.imwrite(dst, face)
                    
                elif emotion == "5":
                    dst = "outputs/val/disgust/" + filename
                    cv2.imwrite(dst, face)
                    
                elif emotion == "6":
                    dst = "outputs/val/anger/" + filename
                    cv2.imwrite(dst, face)
                    
                elif emotion == "7":
                    dst = "outputs/val/contempt/" + filename
                    cv2.imwrite(dst, face)
                    
                elif emotion == "9":
                    dst = "outputs/val/uncertain/" + filename
                    cv2.imwrite(dst, face)
                    
                
                break;
