import cv2
import sys
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import argparse


parser = argparse.ArgumentParser(description='PyTorch Emotion Training- Preprocessing')

parser.add_argument('--modelPath', default="trained_models/ShuffleNet_131.pt", help='path of saved model')

args = parser.parse_args()


cascPath = "data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

classes = ('anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'uncertain')

model = torch.load(args.modelPath)

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    
    _, image = video_capture.read()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(112, 112)
    )

    for (x, y, w, h) in faces:
        
        font_color = (0, 255, 0)
        
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        
        img_tensor = transform(face)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor)
        fc_out = model(img_variable)
        
        emotion_num = fc_out.data.numpy().argmax()
        emotion = classes[emotion_num]
        
        if emotion_num == 0:
            font_color = (0,0,255)
            
        elif emotion_num == 1:
            font_color = (0,0,128)
            
        elif emotion_num == 2:
            font_color = (0,100,0)
            
        elif emotion_num == 3:
            font_color = (130,0,75)
        
        elif emotion_num == 5:
            font_color = (79,79,48)
            
        elif emotion_num == 6:
            font_color = (205,0,0)
            
        elif emotion_num == 7:
            font_color = (0,69,255)
            
        elif emotion_num == 8:
            font_color = (255,255,255)
        
        cv2.rectangle(image, (x, y), (x+w, y+h), font_color, 2)
        cv2.putText(image,emotion,(x, y), cv2.FONT_HERSHEY_SIMPLEX, 3, font_color)


    cv2.imshow('Video', image)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow('Video', image)

video_capture.release()
cv2.destroyAllWindows()
 
