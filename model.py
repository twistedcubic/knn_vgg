
import torch
#import torchvision.models as models
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from vgg import VGG
import torch.nn as nn
from collections import OrderedDict
import os

BATCH_SZ = 512
use_gpu = False

if use_gpu:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = 'cpu'

'''
Replace last part of VGG with knn, compare results
'''
def vgg_knn_compare(model, val_loader, feat_precomp, class_precomp, k=5):
    #load data
    #val_loader = get_loader()
    #model=models.vgg16(pretrained=True)
    #evaluate in batches
    model.eval()
    vgg_cor = 0
    knn_cor = 0
    total = len(val_loader.dataset)
    
    for i, (input, target) in enumerate(val_loader):
        #x_vgg = vgg16(*data)
        #x_knn = feat_extract(vgg16, *data)
        input, target = input.to(device), target.to(device)
        #compare the models
        vgg_cor += vgg_correct(model, input, target) 
        knn_cor += knn_correct(model, k, input, target, feat_precomp, class_precomp)
        print('corr {} {}'.format(vgg_cor, knn_cor))
        del input, target
    vgg_acc = vgg_cor/total
    knn_acc = knn_cor/total
    print('vgg_acc: {} knn_acc: {}'.format(vgg_acc, knn_acc))
    
    return vgg_acc, knn_acc

'''
Input:
-x, img data, in batches
'''
def vgg_correct(model, input, target):
    #for i, (input, target) in enumerate(dataloader):
    output = model(input)
    _, output = output.max(1)
    
    
    #print('vgg correct , output {}  {}'.format(output, target))
    correct = output.eq(target).sum(0)

    return correct

'''
Applies model to get features, find nearest amongst feature2img.
Input:
-x, img data, in batches.
'''
def knn_correct(model, k, input, targets, feat_precomp, class_precomp):
    
    embs = feat_extract(model, input)
    #print('embs size {}'.format(embs.size()))
    correct = 0
    dist_func = nn.PairwiseDistance()
    
    #find nearest for each
    for i, emb in enumerate(embs):
        #feat_precomp shape (batch_sz, 512), emb shape (512)
        dist = dist_func(emb.unsqueeze(0), feat_precomp)
        target = targets[i]
        val, idx = torch.topk(dist, k, largest=False)
        #get classes of the imgs at idx
        pred_classes = class_precomp[idx]
        #print('idx  target {} {}'.format(idx, pred_classes))
        
        if target in pred_classes: ####
            correct += 1
    return correct
    
'''
Extract features from trained model using input data. 
Inputs:
-model: Pytorch pretrained model.
-x: img data, in batch
Returns:
-embedding of input data
'''
def feat_extract(model, x):
    model.eval()
    #right now replace the classifier part with kNN
    #shape (batch_sz, 512, 1, 1)
    #x = model.module.features(x)
    x = model.features(x)
    
    x = x.view(x.size(0), x.size(1))
    #print('x size {}'.format(x.size()))
    return x
    
'''
Utilities functions to create feature data points on input images
for kNN
'''
'''
Returns embeddings of imgs in dataset, as lists of data and embeddings.
Input:
-knn: vgg model
'''
def create_embed(model, dataloader):
    embs = None
    
    targets = torch.LongTensor(len(dataloader.dataset))
    #datalist = []
    counter = 0
    batch_sz = 0
    for i, (input, target) in enumerate(dataloader):
        #input and target have sizes e.g. torch.Size([256, 3, 32, 32])  torch.Size([256]) 
        #print('input tgt sizes {}  {}'.format(input.size(), target.size()))
        input, target = input.to(device), target.to(device)
        #size (batch_sz, 512)
        feat = feat_extract(model, input)
        
        if embs is None:
            batch_sz = target.size(0)
            embs = torch.FloatTensor(len(dataloader.dataset), feat.size(1)) #########
            #targets = None##########
        #print('{} feat sz {}'.format(embs.size(), feat.size()))
        embs[counter:counter+feat.size(0), :] = feat
        targets[counter:counter+feat.size(0)] = target
        counter += batch_sz
        #check loader tensor or list########
        #datalist.extend(input)

    return embs, targets

transforms_train = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]

transforms_test = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
'''
Create dataloader
'''
def get_loader(train=False):
    if train:
        transf = transforms_train
    else:
        transf = transforms_test
    #use cifar10 built-in dataset
    data_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='data', train=train,
                         transform=transforms.Compose(
                             transf                             
                         ), download=True
        ),
        batch_size=BATCH_SZ, shuffle=False
        
        )
    return data_loader

'''
path: path to check point
'''
def get_model(path='pretrained/ckpt.t7'):
    #model = nn.DataParallel(VGG('VGG16'))
    model = VGG('VGG16')
    
    #checkpoint=torch.load(path, map_location='cpu')
    #checkpoint=torch.load(path, map_location=lambda storage, loc:storage)
    state_dict = torch.load(path)['net']
    new_dict = OrderedDict()
    for key, val in state_dict.items():
        key = key[7:] #since 'module.' has len 7
        new_dict[key] = val.to('cpu')
    model.load_state_dict(new_dict)
    return model

if __name__ == '__main__':
    # train_loader = get_loader(train=True)
        
    model = get_model()
    embs_path = 'data/train_embs.pth'
    targets_path = 'data/train_classes.pth'
    if os.path.isfile(embs_path) and os.path.isfile(targets_path):
        embs = torch.load(embs_path)
        targets = torch.load(targets_path)
    else:        
        train_loader = get_loader(train=False)
        #compute feat embed for training data. (total_len, 512)
        embs, targets = create_embed(model, train_loader)
        torch.save(embs, 'data/train_embs.pth')
        torch.save(targets, 'data/train_classes.pth')
    
    val_loader = get_loader(train=False)
    vgg_acc, knn_acc = vgg_knn_compare(model, val_loader, embs, targets, k=5)
    
    
