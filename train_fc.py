
'''
Train fully connected last layer, fixing weights of truncated net.
'''
import torch
import torch.optim as optim
import torch.nn as nn
import pdb
import os

import model

use_gpu = model.use_gpu
device = model.device

class FCL():
    '''
    Input:
    -net: truncated net, weights fixed
    -num_layers: number of layers to peel
    '''
    def __init__(self, net, num_layers):
        
        self.n_out = 512
        self.num_layers = num_layers
        if num_layers >= 21:
            self.n_out = 256 
        self.fc = nn.Linear(self.n_out, 10)
        self.pretrained = net

    '''
    Train fc and save
    Input: path to save
    '''
    def train_fc(self, trainloader, testloader, epochs=50, path='pretrained'):
        print('training for peeling off {} layers'.format(self.num_layers))
        optimizer = optim.SGD(self.fc.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4) #######
        criterion = nn.CrossEntropyLoss()
        best_acc = 0
        #every number of epochs to test
        test_interval = 10
        test_total = len(testloader.dataset)

        if use_gpu:
            self.fc.cuda()
            self.pretrained.cuda()
            
        self.pretrained.eval()
        for epoch in range(epochs):
            for i, (input, target) in enumerate(trainloader):
                if use_gpu:
                    input = input.to(device)
                    target = target.to(device)
                with torch.no_grad():                    
                    x = self.pretrained(input).squeeze()
                #print('x size {}'.format(x.size()))    
                x = self.fc(x)
                optimizer.zero_grad()
                loss = criterion(x, target)
                loss.backward()
                optimizer.step()
                
            if (epoch + 1) % test_interval == 0:
                correct = 0
                #pdb.set_trace()
                with torch.no_grad():
                    for i, (input, target) in enumerate(testloader):
                        input = input.to(device)
                        target = target.to(device)
                        x = self.pretrained(input).squeeze()
                        x = self.fc(x)
                        cls = x.max(dim=1)[1]
                        correct += cls.eq(target).sum()
                        
                acc = int(correct)/test_total
                #print('acc {} '.format(acc))
                if epoch > 20 and acc > best_acc:
                    print('best acc after peeling off num_layers {}: {}'.format(self.num_layers, acc))
                    save_path = os.path.join(path, 'fc'+str(self.num_layers) + '.t7')
                    torch.save(self.fc.state_dict(), save_path)
                    best_acc = acc
                    

if __name__ == '__main__':
    #this model has all layers
    net = model.get_model()
    num_layers_l = [5, 8, 11, 15, 18, 21, 25, 28]
    num_layers_l = [11, 15, 18, 21, 25, 28]
    #num_layers_l = [8]
    
    trainloader = model.get_loader(train=True)
    testloader = model.get_loader(train=False)

    for num_layers in num_layers_l:
        cur_model = model.peel_layers(net, num_layers)
        fcl = FCL(cur_model, num_layers)
        fcl.train_fc(trainloader, testloader, epochs=50)
