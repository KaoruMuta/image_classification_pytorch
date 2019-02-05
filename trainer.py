#!/bin/python
from optparse import OptionParser
import os
import shutil
import numpy as np
import itertools
import torch
from torch.autograd import Variable
from random import shuffle
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, auc, roc_curve
# Custom imports
from Dataloader import *
import pretrainedmodels
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(options):
    #Warning if the indicated directory name is already existed
    if os.path.exists(options.outputDir):
        print('This directory is already existed!')
        exit(-1)

    os.mkdir(options.outputDir)
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    predictedLabels = []
    gtLabels = []
    best_acc = 0.0

    # Create model
    if options.model == 'resnet':
        print('model')
        model = models.resnet152(pretrained=True)
        '''
        #if using feature extracting
        for param in model.parameters():
            #param.requires_grad = False
        '''
        # Identify the name of the last layer
        for name, child in model.named_children():
            for name2, params in child.named_parameters():
                print(name, name2)

        ## Change the last layer
        inputDim = model.fc.in_features
        model.fc = torch.nn.Linear(inputDim, options.numClasses)
        # TODO: Rectify the transform params
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_size = 224

    elif options.model == 'NAS':
        # Use pretrained models library -  (pip install --upgrade pretrainedmodels)
        # https://github.com/Cadene/pretrained-models.pytorch
        model = pretrainedmodels.__dict__['nasnetalarge'](num_classes=1000, pretrained='imagenet')

        # Change the last layer
        inputDim = model.last_linear.in_features
        model.last_linear = torch.nn.Linear(inputDim, options.numClasses)

        mean = model.mean
        std = model.std
        input_size = model.input_size[1]
        print('mean:', mean)
        print('std:', std)
        print('input size:', input_size)

        assert model.input_size[1] == model.input_size[2], 'Error: Models expects different dimensions for height and width'
        assert model.input_space == 'RGB', 'Error: Data loaded in RGB format while the model expects BGR'

    #Multiple GPU used
    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs!')
        model = torch.nn.DataParallel(model)

    # Move the model to desired device
    model.to(device)

    #RandomResizedCrop and RandomHorizontalFlip preprocessing
    dataTransform = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    #FiveCrop preprocessing
    dataTransformVal = transforms.Compose([
        transforms.Resize(input_size),
        transforms.FiveCrop(input_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=mean, std=std)(crop) for crop in crops]))])

    print('loading train dataset')
    trainset = MyDataset(options.rootDir, split=Data.TRAIN, transform=dataTransform)
    trainLoader = DataLoader(dataset=trainset, num_workers=0, batch_size=options.batchSize, shuffle=True)
    assert options.numClasses == trainset.getNumClasses(), 'Error: Number of classes found in the dataset is not equal to the number of classes specified in the options (%d != %d)!' % (trainset.getNumClasses(), options.numClasses)

    print('loading validation dataset')
    valset = MyDataset(options.rootDir, split=Data.VAL, transform=dataTransformVal)
    valLoader = DataLoader(dataset=valset, num_workers=0, batch_size=options.batchSize, shuffle=False)
    assert options.numClasses == valset.getNumClasses(), 'Error: Number of classes found in the dataset is not equal to the number of classes specified in the options (%d != %d)!' % (valset.getNumClasses(), options.numClasses)

    # Define optimizer and scheduler (scheduler is just optional)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(options.trainingEpochs):
        model.train()
        train_loss = 0.0
        print('epoch start')
        # Start training
        for iterationIdx, data in enumerate(trainLoader):
            X = data['data']
            y = data['label']
            # Move the data to PyTorch on the desired device
            X = Variable(X).float().to(device)
            y = Variable(y).long().to(device)
            # Get model predictions
            pred = model(X)
            # Optimize
            optimizer.zero_grad()
            loss = criterion(pred, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(pred.data, dim = 1)
            #From Tensor to Numpy
            predictedLabels.append(preds.cpu().numpy())
            gtLabels.append(y.data.cpu().numpy())
            if iterationIdx % options.displayStep == 0:
                print('Epoch %d | Iteration: %d | Loss: %.5f' % (epoch, iterationIdx, loss))

        print('train_Loss:', train_loss / options.batchSize)
        with open(os.path.join(options.outputDir, 'log_trainloss.txt'), 'a') as log_trainloss:
            print(train_loss / options.batchSize, file=log_trainloss)

        #From 2d ndarray to 1d ndarray
        predictedLabels = predictedLabels.ravel()
        gtLabels = gtLabels.ravel()

        print('Size of predictedLabels:', predictedLabels.shape[0])
        print('Size of gtLabels:', gtLabels.shape[0])
        epoch_acc = accuracy_score(gtLabels, predictedLabels)
        print('train_acc:', epoch_acc)
        with open(os.path.join(options.outputDir, 'log_trainacc.txt'), 'a') as log_trainacc:
            print(epoch_acc, file=log_trainacc)

        val_loss = 0.0
        #Redefined two labels again
        predictedLabels = []
        gtLabels = []
        model.eval()
        for iterationIdx, data in enumerate(valLoader):
            X = data['data']
            y = data['label']
            # Move the data to PyTorch on the desired device
            X = Variable(X).float().to(device)
            y = Variable(y).long().to(device)
            #testing the dataset
            bs, ncrops, c, h, w = X.size()
            with torch.no_grad():
                temp_output = model(X.view(-1, c, h, w))
            outputs = temp_output.view(bs, ncrops, -1).mean(1)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            _, preds = torch.max(outputs.data, dim = 1)
            #converting tensor to numpy
            predictedLabels.append(preds.cpu().numpy())
            gtLabels.append(y.data.cpu().numpy())

        print('val_Loss:', val_loss / options.batchSize)
        with open(os.path.join(options.outputDir, 'log_valloss.txt'), 'a') as log_valloss:
            print(train_loss / options.batchSize, file=log_valloss)

        gtLabels = gtLabels.ravel()
        predictedLabels = predictedLabels.ravel()

        epoch_acc = accuracy_score(gtLabels, predictedLabels)
        print('val_acc:', epoch_acc)
        with open(os.path.join(options.outputDir, 'log_valacc.txt'), 'a') as log_valacc:
            print(epoch_acc, file=log_valacc)

        scheduler.step()
        if best_acc < epoch_acc:
            best_acc = epoch_acc
            print('Updating best model!')
            # Save model
            torch.save(model.state_dict(), os.path.join(options.outputDir, 'best_model.pth'))

def test(options):
    #Warning if the indicated directory name is already existed
    if not os.path.exists(options.outputDir):
        print('This directory is not existed!')
        exit(-1)

    # Create model
    if options.model == 'resnet':
        print('model')
        model = models.resnet152(pretrained=True)
        '''
        #if using feature extracting
        for param in model.parameters():
            #param.requires_grad = False
        '''
        # Identify the name of the last layer
        for name, child in model.named_children():
            for name2, params in child.named_parameters():
                print(name, name2)

        ## Change the last layer
        inputDim = model.fc.in_features
        model.fc = torch.nn.Linear(inputDim, options.numClasses)
        # TODO: Rectify the transform params
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_size = 224

    elif options.model == 'NAS':
        # Use pretrained models library -  (pip install --upgrade pretrainedmodels)
        # https://github.com/Cadene/pretrained-models.pytorch
        model = pretrainedmodels.__dict__['nasnetalarge'](num_classes=1000, pretrained='imagenet')

        # Change the last layer
        inputDim = model.last_linear.in_features
        model.last_linear = torch.nn.Linear(inputDim, options.numClasses)

        mean = model.mean
        std = model.std
        input_size = model.input_size[1]
        print('mean:', mean)
        print('std:', std)
        print('input size:', input_size)

        assert model.input_size[1] == model.input_size[2], 'Error: Models expects different dimensions for height and width'
        assert model.input_space == 'RGB', 'Error: Data loaded in RGB format while the model expects BGR'

    #Multiple GPU used
    if torch.cuda.device_count() > 1:
        print('Using', torch.cuda.device_count(), 'GPUs!')
        model = torch.nn.DataParallel(model)

    # Move the model to desired device
    model.to(device)

    #FiveCrop preprocessing
    dataTransformVal = transforms.Compose([
        transforms.Resize(input_size),
        transforms.FiveCrop(input_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean=mean, std=std)(crop) for crop in crops]))])

    print('loading test dataset')
    testset = MyDataset(options.rootDir, split=Data.TEST, transform=dataTransformVal)
    testLoader = DataLoader(dataset=testset, num_workers=0, batch_size=options.batchSize, shuffle=False)
    assert options.numClasses == testset.getNumClasses(), 'Error: Number of classes found in the dataset is not equal to the number of classes specified in the options (%d != %d)!' % (testset.getNumClasses(), options.numClasses)

    print('Model restored!')
    modelCheckpoint = torch.load(os.path.join(options.outputDir, 'best_model.pth'))
    model.load_state_dict(modelCheckpoint) #Loading weights of best model
    correctExamples = 0
    predictedLabels = []
    gtLabels = []
    oneHot = []
    pred_fold = []
    model.eval()
    for iterationIdx, data in enumerate(testLoader):
        X = data['data']
        y = data['label']
        # Move the data to PyTorch on the desired device
        X = Variable(X).float().to(device)
        y = Variable(y).long().to(device)
        #testing the dataset
        bs, ncrops, c, h, w = X.size()
        with torch.no_grad():
            temp_output = model(X.view(-1, c, h, w))
        outputs = temp_output.view(bs, ncrops, -1).mean(1)

        _, preds = torch.max(outputs.data, dim = 1)
        correctExamples += (preds == y.data).sum().item()
        #converting tensor to numpy
        predictedLabels.append(preds.cpu().numpy())
        gtLabels.append(y.data.cpu().numpy())
        m = torch.nn.Softmax(dim = 1)
        #Outputs to probability
        outputs = m(outputs)
        #From Tensor to Numpy
        outputs = outputs.cpu().numpy()
        for i in range(len(outputs)):
            pred_fold.append(outputs[i])

    predictedLabels = predictedLabels.ravel()
    gtLabels = gtLabels.ravel()

    #Producing oneHot label
    for i in range(len(gtLabels)):
        oh = np.zeros(options.numClasses)
        oh[gtLabels[i]] = 1.0
        oneHot.append(oh)

    #From list to Numpy
    oneHot = np.asarray(oneHot)
    pred_fold = np.asarray(pred_fold)
    Classes=[i for i in range(options.numClasses)] #optional
    accuracy = accuracy_score(gtLabels, predictedLabels)
    print('Correct examples: %d | Total examples: %d | Accuracy: %.5f' % (correctExamples, len(predictedLabels), float(correctExamples) / len(predictedLabels)))
    print(classification_report(gtLabels, predictedLabels))
    print('accuracy:', accuracy)
    cnf_matrix = confusion_matrix(gtLabels, predictedLabels)
    #plotting comfusion matrix
    plot_cfmatrix(cnf_matrix, classes=Classes, title='Confusion matrix', cmap=plt.cm.RdPu)
    #In multiple class, evaluating AUC score
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    #
    for i in range(options.numClasses):
        fpr[i], tpr[i], _ = roc_curve(oneHot[:, i], pred_fold[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr['micro'], tpr['micro'], _ = roc_curve(oneHot.ravel(), pred_fold.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    print('AUC score:', roc_auc['micro'])
    plt.figure()
    plt.plot(fpr['micro'], tpr['micro'], label='Micro Averaging ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(options.outputDir, 'roc_curve.png'))

    with open(os.path.join(options.outputDir, 'result.txt'), 'a') as inference:
        print(accuracy, file=inference)
        print(roc_auc['micro'], file=inference)
        print(classification_report(gtLabels, predictedLabels), file=inference)

def plot(options): #if you wanna plot, please customize this code
    tloss_x = []
    tloss_y = []
    t_accx = []
    t_accy = []
    with open(os.path.join(options.outputDir, 'log_trainloss.txt'), 'r') as plot_trainloss:
        for idx, loss in enumerate(plot_trainloss):
            tloss_x.append(idx)
            tloss_y.append(float(loss))

    plt.title('train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    #plot the value of loss
    plt.plot(tloss_x, tloss_y, label='train_loss')
    plt.xticks(xlabels, xlabels)
    plt.legend()
    plt.savefig(os.path.join(options.outputDir, 'loss.png'))

    with open(os.path.join(options.outputDir, 'train_acc.txt'), 'r') as plot_trainacc:
        for idx, loss in enumerate(plot_trainacc):
            t_accx.append(idx)
            t_accy.append(float(loss))

    plt.figure()
    plt.title('train accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    #plot the value of accuracy
    plt.plot(t_accx, t_accy, label='train_acc')
    plt.xticks(xlabels, xlabels)
    plt.legend()
    plt.savefig(os.path.join(options.outputDir, 'accuracy.png'))

def plot_cfmatrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.RdPu): #if you wanna plot confusion matrix, please use it
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[np:newaxis]
        print('Normlized confusion matrix')
    else:
        print('confusion matrix')

    print(cm)

    plt.figure(figsize=(12,12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(options.outputDir, 'confusion_matrix.png'))

if __name__ == '__main__':
    # Command line options
    parser = OptionParser()

    # Base options
    parser.add_option('-m', '--model', action='store', type='string', dest='model', default='NAS', help='Model to be used for Cross-Layer Pooling')
    parser.add_option('-t', '--trainModel', action='store_true', dest='trainModel', default=False, help='Train model')
    parser.add_option('-c', '--testModel', action='store_true', dest='testModel', default=False, help='Test model')
    parser.add_option('-o', '--outputDir', action='store', type='string', dest='outputDir', default='./output', help='Output directory')
    parser.add_option('-e', '--trainingEpochs', action='store', type='int', dest='trainingEpochs', default=10, help='Number of training epochs')
    parser.add_option('-b', '--batchSize', action='store', type='int', dest='batchSize', default=32, help='Batch Size')
    parser.add_option('-d', '--displayStep', action='store', type='int', dest='displayStep', default=2, help='Display step where the loss should be displayed')
    parser.add_option('-p', '--plot', action='store_true', dest='plot', default=False, help='plot')

    # Input Reader Params
    parser.add_option('--rootDir', action='store', type='string', dest='rootDir', default='../data/', help='Root directory containing the data')
    parser.add_option('--numClasses', action='store', type='int', dest='numClasses', default=7, help='Number of classes in the dataset')
    parser.add_option('--lr', action='store', type='float', dest='lr', default=0.0001, help='the value of learning rate')
    # Parse command line options
    (options, args) = parser.parse_args()
    print(options)

    if options.trainModel:
        print('Training model')
        train(options)

    if options.testModel:
        print('Testing model')
        test(options)

    if options.plot:
        print('plot now')
        plot(options)
