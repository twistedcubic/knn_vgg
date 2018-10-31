
import torch
#import torchvision.models as models
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from vgg import VGG
import torch.nn as nn
from collections import OrderedDict
import os
import time

BATCH_SZ = 128
use_gpu = True
normalize_feat = True

if use_gpu:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = 'cpu'

'''
Replace last part of VGG with knn, compare results
'''
def vgg_knn_compare(model, val_loader, feat_precomp, class_precomp, k=5, normalize_feat=False, num_layers=None):
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
        with torch.no_grad():
            vgg_cor += vgg_correct(model, input, target) 
            knn_cor += knn_correct(model, k, input, target, feat_precomp, class_precomp, normalize_feat=normalize_feat, num_layers=num_layers)
        #print('corr {} {}'.format(vgg_cor, knn_cor))
        
        del input, target
    vgg_acc = vgg_cor/total
    knn_acc = knn_cor/total
    print('Peeling off {} layers -- vgg_cor: {} knn_cor: {} total: {}'.format(num_layers, vgg_cor, knn_cor, total))
    print('knn_acc: {}'.format(knn_acc))
    
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
def knn_correct(model, k, input, targets, feat_precomp, class_precomp, normalize_feat=False, num_layers=None):

    model = peel_layers(model, num_layers)
    #print('peeled_model {}'.format(model))
    embs = model(input)
    
    embs = embs.view(embs.size(0), embs.size(1))    
    #embs = feat_extract(model, input, num_layers=num_layers)
    
    #print('embs size {}'.format(embs.size()))
    correct = 0
    dist_func_name = 'l2'
    #if dist_func_name == 'l2':
    dist_func = nn.PairwiseDistance()

    #can run matmul on entire batch instead of one!
    #find nearest for each
    for i, emb in enumerate(embs):
        
        #feat_precomp shape (batch_sz, 512), emb shape (512)
        if dist_func_name == 'l2':
            dist = dist_func(emb.unsqueeze(0), feat_precomp)
        else:
            if normalize_feat:
                emb = F.normalize(emb, p=2, dim=0)
            dist = torch.matmul(emb.unsqueeze(0), feat_precomp.t()).view(feat_precomp.size(0))
            #print('dist {}'.format(dist.size()))
            
        val, idx = torch.topk(dist, k, largest=(False if dist_func_name=='l2' else True))
        
        target = targets[i]
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
-num_layers: num_layers to peel
Returns:
-embedding of input data
'''
def feat_extract(model, x, num_layers=None):
    model.eval()
    #right now replace the classifier part with kNN
    #shape (batch_sz, 512, 1, 1)
    if peel_layers != None:
        cur_model = peel_layers(model, num_layers)
    else:
        if use_gpu:
            cur_model = model.module.features
        else:
            cur_model = model.features
    #print('peeled model: {}'.format(cur_model))
    x = cur_model(x)
    '''
    if use_gpu:
        x = model.module.features(x)
        #x = cur_model(x)
    else:
        x = model.features(x)
    '''
    #print('x size {}'.format(x.size()))
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
def create_embed(model, dataloader, normalize_feat=True, num_layers=None):
    model.eval()
    embs = None
    #conserves memory
    stream = False
    targets = torch.LongTensor(len(dataloader.dataset)).to(device)    
    #datalist = []
    file_counter = 0
    counter = 0
    batch_sz = 0
    for i, (input, target) in enumerate(dataloader):
        #input and target have sizes e.g. torch.Size([256, 3, 32, 32])  torch.Size([256]) 
        #print('input tgt sizes {}  {}'.format(input.size(), target.size()))
        input, target = input.to(device), target.to(device)
        #size (batch_sz, 512)
        with torch.no_grad():
            feat = feat_extract(model, input, num_layers=num_layers)
        
        if not stream:
            if embs is None:
                batch_sz = target.size(0)            
                embs = torch.FloatTensor(len(dataloader.dataset), feat.size(1)).to(device)
            embs[counter:counter+feat.size(0), :] = feat
            targets[counter:counter+feat.size(0)] = target
            counter += batch_sz
        elif stream:
            embs = torch.FloatTensor(feat.size(0), feat.size(1))
            torch.save(embs, 'data/train_embs{}.pth'.format(file_counter))
            torch.save(target, 'data/train_classes{}.pth'.format(file_counter))
            file_counter += 1
        
        del input, target, feat
        #print('{} feat sz {}'.format(embs.size(), feat.size()))
        if normalize_feat:
            embs = F.normalize(embs, p=2, dim=1)
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
Create module with given number of blocks. Used for training after layers are
peeled off.
Input:
-model to use
-num_layers, number of layers in the vgg Sequential component to peel off. Should be multiples of 3 for each block, then plus 1
note the final avg pool layer should always be included in the end. Count max pool layer within the peeled off layers.
'''
def peel_layers(model, num_layers):
    if num_layers is None:
        if use_gpu:
            return model.module.features
        else:
            return model.features
    #modules have name .features, which is a Sequential, within which certain blocks are picked.
    # 'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    #model._modules is OrderedDict
    if use_gpu:
        model_layers = model._modules['module']._modules['features']._modules
    else:
        model_layers = model._modules['features']._modules
    total_layers = len(model_layers) - num_layers
    final_layers = []
    #OrderedDict order guaranteed
    for i, (name, layer) in enumerate(model_layers.items()):
        if i == total_layers:
            break
        final_layers.append(layer)
    if num_layers > 11:        
        final_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if num_layers > 21:
            final_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    #append MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), does not include params
    #append AvgPool2d(kernel_size=1, stride=1, padding=0), which does not have params, so no need to copy
    final_layers.extend([nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) , nn.AvgPool2d(kernel_size=1, stride=1)])
    
    return nn.Sequential(*final_layers)
    

'''
path: path to check point
'''
def get_model(path='pretrained/ckpt.t7'):
    #model = nn.DataParallel(VGG('VGG16'))
    model = VGG('VGG16')
    #if True:
    #    print(model._modules['features']._modules)
    #checkpoint=torch.load(path, map_location='cpu')
    #checkpoint=torch.load(path, map_location=lambda storage, loc:storage)
    #model was trained on GPU
    state_dict = torch.load(path)['net']
    if False:
        print(state_dict.keys())
    if False:
        print(list(model._modules.keys()))
        if use_gpu:
            print(model._modules['module']._modules['features']._modules)
        else:
            print(model._modules['features']._modules)
           
    if use_gpu:
        model = nn.DataParallel(VGG('VGG16'))
    else:
        model = VGG('VGG16')
        new_dict = OrderedDict()
        for key, val in state_dict.items():
            key = key[7:] #since 'module.' has len 7
            new_dict[key] = val.to('cpu')
        state_dict = new_dict
    if False:
        print(state_dict.keys())
    model.load_state_dict(state_dict)
    return model

'''
Load features for given number of layers peeled off.
'''
def load_features(num_layers):
    embs_path = 'data/train_embs_norm{}.pth'.format(num_layers)
    targets_path = 'data/train_classes_norm{}.pth'.format(num_layers)
    if os.path.isfile(embs_path) and os.path.isfile(targets_path):
        embs = torch.load(embs_path)
        targets = torch.load(targets_path)
    else:
        raise Exception('Files don\'t exist {}, {}'.format(embs_path, targets_path))
    return embs, targets
   
'''
Input:
-num_layers: number of layers to peel
'''
def run_and_compare(num_layers=None):
    
    print('num_layers to peel: {}'.format(num_layers))
    model = get_model()
    
    model.eval()
    if normalize_feat:
        embs_path = 'data/train_embs_norm{}.pth'.format(num_layers)
        targets_path = 'data/train_classes_norm{}.pth'.format(num_layers)
    else:
        embs_path = 'data/train_embs.pth'
        targets_path = 'data/train_classes.pth'

    #set to False for fast testing.
    training=True
    train_loader = get_loader(train=training)
    if os.path.isfile(embs_path) and os.path.isfile(targets_path):
        embs = torch.load(embs_path)
        targets = torch.load(targets_path)
    else:        
        #compute feat embed for training data. (total_len, 512)
        embs, targets = create_embed(model, train_loader, normalize_feat=normalize_feat, num_layers=num_layers)
        #should save as two-element dict
        torch.save(embs, embs_path)
        torch.save(targets, targets_path)
    
    print('Done creating or loading embeddings for knn!')
    val_loader = get_loader(train=False)
    print('train dataset size {} test dataset size {}'.format(len(train_loader.dataset), len(val_loader.dataset) ))
    
    vgg_acc, knn_acc = vgg_knn_compare(model, val_loader, embs, targets, k=1, normalize_feat=normalize_feat, num_layers=num_layers)
    
    
if __name__ == '__main__':
    # train_loader = get_loader(train=True)        
    
    #should be in multiples of 3, then plus 2, and plus however many M is there
    allowed_peel = [5, 8, 11, 15, 18, 21, 25, 28] #from 21 on, 256 dim instead of 512
    #number of layers to peel
    num_layers_l = [15, 18, 21, 25, 28]#, 15, 18, 22, 25, 28]
    for num_layers in num_layers_l:
        run_and_compare(num_layers=num_layers)
