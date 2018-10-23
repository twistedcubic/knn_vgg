
import torch
#import torchvision.models as models
import torch.nn.functional as F
import torchvision.datasets as datasets

'''
Replace last part of VGG with knn, compare results
'''
def vgg_knn_compare(model, val_loader, feat_precomp, class_precomp, k=5):
    #load data
    #val_loader = get_loader()
    #model=models.vgg16(pretrained=True)
    #evaluate in batches
    model.eval()
    vgg_correct = 0
    knn_correct = 0
    total = len(val_loader)
    
    for i, (input, target) in enumerate(val_loader):
        #x_vgg = vgg16(*data)
        #x_knn = feat_extract(vgg16, *data)
        #compare the models
        vgg_correct += vgg_correct(model, *data  ) 
        knn_correct += knn_correct(model, k, *data, feat_precomp, class_precomp)
    vgg_acc = vgg_correct//total
    knn_acc = knn_correct//total
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
    #add batch size
    #total = target.size(0)
    correct = output.eq(target).sum(0)

    return correct

'''
Applies model to get features, find nearest amongst feature2img.
Input:
-x, img data, in batches.
'''
def knn_correct(model, k, input, target, feat_precomp, class_precomp):
    
    embs = feat_extract(model, input)
    correct = 0
    dist_func = nn.PairwiseDistance()
    
    #find nearest for each
    for emb in embs:
        dist = dist_func(emb.unsqueeze(0), feat_precomp)
        
        val, idx = torch.topk(dist, k, largest=False)
        if target in idx: ####
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
    x = model.features(x)
    #reshape?
    
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
    targets = None
    #datalist = []
    counter = 0
    batch_sz = 0
    for i, (input, target) in enumerate(dataloader):
        if embs == None:
            batch_sz = target.size(0)
            embs = torch.FloatTensor(dataloader.size(), target.size(1), target.size(2) ) #########
            targets = 
        emb[counter:counter+batch_sz, :, :] = feat_extract(model, input)
        
        counter += batch_sz
        #check loader tensor or list########
        #datalist.extend(input)

    return embs, targets

'''
Create dataloader
'''
def get_loader(train=False):
    
    #use cifar10 built-in dataset
    data_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='data', train=train,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                         ]), download=True
        ),
        batch_size=opt.batch_size, shuffle=False
        
        )
    return data_loader

'''
path: path to check point
'''
def get_model(path=' '):
    model = VGG16()
    checkpoint=torch.load(path)
    model.load_state_dict(checkpoint['net'])
    return model

if __name__ == '__main__':
    train_loader = get_loader(train=True)
    
    model = get_model() 
    #compute feat embed for training data
    embs, targets = create_embed(model, train_loader)
    val_loader = get_loader(train=False)
    vgg_acc, knn_acc = vgg_knn_compare(val_loader, embs, targets, k=5)
    
    
