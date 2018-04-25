import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from shufflenet import ShuffleNet
from mobilenet import MobileNet
from torchsummary import summary
import time


parser = argparse.ArgumentParser(description='PyTorch Emotion Training')
parser.add_argument('--model', default='ShuffleNet', metavar='M', help='model: ShuffleNet or MobileNet')

args = parser.parse_args()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

trainset = torchvision.datasets.ImageFolder(root='./outputs/train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=32, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='./outputs/val', transform=transform)
testloader = torch.utils.data.DataLoader(testset, shuffle=False, batch_size=4, num_workers=2)

classes = ('anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'uncertain')

model = ShuffleNet()

if args.model == 'MobileNet' :
    model = MobileNet()


summary(model, (3,224,224))
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

max_accuracy = 0.0


for epoch in range(90):  # loop over the dataset multiple times
    
    model.train(True)

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 5 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
            
        if i == 25 :
            break


    print('finished epoch %d' % epoch)
    print('calculating accuracy')
    
    start_time = time.time()


    model.train(False)
    
    correct = 0
    total = 0.0001
    for data in testloader:
        images, labels = data
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        
        # your code
    elapsed_time = time.time() - start_time
    accuracy = (100 * correct / total)
    print('Accuracy of the network on the test images: %d %%' % accuracy)
    print('Took %d seconds to validate' % elapsed_time)

    
    
    
    if accuracy > max_accuracy :
        torch.save(model, args.model + ".pt")
        max_accuracy = accuracy



    class_correct = list(0. for i in range(9))
    class_total = list(0. for i in range(9))
    for data in testloader:
        images, labels = data
        #print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        #print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1


    for i in range(9):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        


