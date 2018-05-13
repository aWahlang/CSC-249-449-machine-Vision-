import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
from shufflenet2 import ShuffleNet
from mobilenet import MobileNet
#from torchsummary import summary
import time


parser = argparse.ArgumentParser(description='PyTorch Emotion Training- Preprocessing')

parser.add_argument('--modelPath', default="trained_models/ShuffleNet_130.pt", help='path of saved model')


args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("device: " + str(device))
    

model = torch.load(args.modelPath, map_location=str(device))

print("loading data...")


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


testset = torchvision.datasets.ImageFolder(root='./outputs/val', transform=transform)
testloader = torch.utils.data.DataLoader(testset, shuffle=True, batch_size=1)

classes = ('anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')


model.train(False)


correct = 0
total = 1
count = 0

class_correct = list(0. for i in range(8))
class_total = list(1. for i in range(8))
class_map = np.zeros((8, 8))

start_time = time.time()

for data in testloader:
    count +=1
    image, label = data


    image, label = image.to(device), label.to(device)
    output = model(Variable(image))
    _, predicted = torch.max(output.data, 1)
    total += 1
    class_total[label[0]] +=1
    
    class_map[label[0]][predicted[0]] += 1

    if (label[0] == predicted[0]):
        correct += 1
        class_correct[predicted[0]] += 1
    
    #if count > 1000:
        #break
        
    if count % 100 == 0:
        
        print('%2d / 2856' % count)
        


elapsed_time = time.time() - start_time
accuracy = (100 * correct / total)
print('Accuracy of the network on the test images: %d%%' % accuracy)
print('Took %d seconds to validate' % elapsed_time)

for i in range(8):
    print('Accuracy of {0} : {1}'.format(classes[i], 100 * class_correct[i] / class_total[i]))

for i in range(8):
    print()
    e_total = class_map[i].sum()
    
    if e_total < 1:
        e_total = 1
        
    print(classes[i] + ":")
    print('anger: {0}%  contempt: {1}%  disgust: {2}%  fear: {3}%  happy: {4}%  neutral: {5}%  sad: {6}%  surprise: {7}%'.format(class_map[i][0]/e_total*100, 
                                                                                                                         (class_map[i][1]/e_total)*100, 
                                                                                                                         (class_map[i][2]/e_total)*100, 
                                                                                                                         (class_map[i][3]/e_total)*100, 
                                                                                                                         (class_map[i][4]/e_total)*100, 
                                                                                                                         (class_map[i][5]/e_total)*100, 
                                                                                                                         (class_map[i][6]/e_total)*100, 
                                                                                                                         (class_map[i][7]/e_total)*100))


    

