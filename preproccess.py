import csv
import cv2
import re
import os.path
import argparse
import random

parser = argparse.ArgumentParser(description='PyTorch Emotion Training- Preprocessing')

parser.add_argument('--train', default="true", help='train: true or false')
parser.add_argument('--csv', default="training.csv", help='train: true or false')
parser.add_argument('--data', default="/home/daniel/Documents/affectnet-manual/", help='location of images')
parser.add_argument('--resume', default=-1, help='where to resume')

args = parser.parse_args()

data_csv = args.csv

datapath = args.data

trainpath = "train"

start_count = int(args.resume)
    
with open(data_csv) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    total = 427298
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
    
    
    count = 1
    for row in readCSV:
        
        count += 1
        
        if count < start_count :
            continue
        
        trainpath = "train"

        if args.train == False:
            trainpath = "val"
            
        #if random.random() < 0.2:
            #trainpath = "test"
            
        
        filename = row[0][row[0].index('/')+1:]
        emotion = row[6]
        
            
        src = datapath + filename
        
        
        print(str(count) + "/" + str(total) + "%")
        
       
        if emotion == "0" or emotion == "1" or emotion == "2" or emotion == "3" or emotion == "6" or emotion == "8" or emotion == "9" or emotion == "10":
            continue
        
        
        if os.path.exists(src) and src.endswith(".jpg"):
            image = cv2.imread(src)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            faces = face_cascade.detectMultiScale(gray,
                                                  scaleFactor=1.2,
                                                  minNeighbors=5,
                                                  minSize=(112, 112))
            for (x,y,w,h) in faces:
                face = image[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))
                
                face = cv2.flip(face, 1)
        
                if emotion == "0":
                    dst = "outputs/" + trainpath + "/neutral/rev-" + filename
                    cv2.imwrite(dst, face)
                    
                elif emotion == "1":
                    dst = "outputs/" + trainpath + "/happy/rev-" + filename
                    cv2.imwrite(dst, face)
                    
                elif emotion == "2":
                    dst = "outputs/" + trainpath + "/sad/rev-" + filename
                    cv2.imwrite(dst, face)
                    
                elif emotion == "3":
                    dst = "outputs/" + trainpath + "/surprise/rev-" + filename
                    cv2.imwrite(dst, face)
                    
                elif emotion == "4":
                    dst = "outputs/" + trainpath + "/fear/rev-" + filename
                    cv2.imwrite(dst, face)
                    
                elif emotion == "5":
                    dst = "outputs/" + trainpath + "/disgust/rev-" + filename
                    cv2.imwrite(dst, face)
                    
                elif emotion == "6":
                    dst = "outputs/" + trainpath + "/anger/rev-" + filename
                    cv2.imwrite(dst, face)
                    
                elif emotion == "7":
                    dst = "outputs/" + trainpath + "/contempt/rev-" + filename
                    cv2.imwrite(dst, face)
                    
                elif emotion == "9":
                    dst = "outputs/" + trainpath + "/uncertain/rev-" + filename
                    cv2.imwrite(dst, face)
                    
                
                break;
