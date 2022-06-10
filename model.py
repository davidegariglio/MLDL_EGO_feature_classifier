# imports
import torch
import pickle as pkl
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from TRNmodule import RelationModuleMultiScaleWithClassifier
from args import parser
from data.dataset import PersonalizedDataset

if __name__ == "__main__":

    args = parser.parse_args()

    num_frames = 5 # I3D vs TSM
    num_class = 8 # 8????
    img_feature_dim = 2048 # 1024 i3d, 2048 tsm

    #if trm oppure avg
    trm = RelationModuleMultiScaleWithClassifier(img_feature_dim, num_frames, num_class)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_features_path = f'/content/drive/MyDrive/MLDL_2022/project/EGO_project/Pre-extracted_feat/{args.modality}/ek_{args.backbone}/D{args.source}-D{args.source}_train.pkl'

    # *** TRAIN pkls***

    with open(source_features_path, "rb") as f:
        p = pkl.load(f)

    train_input_feat = p['features']['RGB']
    train_input_feat= torch.from_numpy(train_input_feat).type(torch.float32)
    train_input_feat.to(device)
    trm.to(device)
    train_input_ids = []
    # for identifier in p['narration_ids']:
    #     train_input_ids.append(identifier.split('_')[-1])
    # extract source labels in order to fit the classifier

    train_labels_path = f'/content/drive/MyDrive/MLDL_2022/project/pkl_files/D{args.source}_train.pkl'

    with open(train_labels_path, "rb") as f:
        p = pkl.load(f)

    df = pd.DataFrame(p)
    labels = df['verb_class'].astype(float)
    labels = labels.to_numpy()
    labels = torch.from_numpy(labels).type(torch.LongTensor)
    labels.to(device)

    # AVG POOL not needed now since we are trying trm
    # data = torch.mean(train_input_feat, 1)
    
    train_dataset = PersonalizedDataset(train_input_feat, labels)
    train_loader = DataLoader(train_dataset, batch_size = 4, shuffle = True)
    # *** TRAINING ***

    # Define the loss
    loss_criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    lr = 0.01
    optim = torch.optim.SGD(trm.parameters(), lr = lr, momentum = 0.9) #stochastic gradient descent

    epochs = 1
    for epoch in range(epochs):
      print(epoch)
      for i, batch_images in enumerate(train_loader):
        feat, lbl = batch_images
        # move images to gpu
        feat, lbl = feat.cuda(), lbl.cuda()

        optim.zero_grad() #at each iteration we zero grad the gradient

        out = trm(feat)

        loss = loss_criterion(out, lbl)
        # backward the error to the model
        loss.backward()
        optim.step()
        
    # *** TEST pkls***

    target_features_path = f'/content/drive/MyDrive/MLDL_2022/project/EGO_project/Pre-extracted_feat/{args.modality}/ek_{args.backbone}/D{args.target}-D{args.target}_test.pkl'

    with open(target_features_path, "rb") as f:
        p = pkl.load(f)

    test_input_feat = p['features']['RGB']
    test_input_feat= torch.from_numpy(test_input_feat).type(torch.float32)
    test_input_feat.to(device)
    print(test_input_feat.is_cuda)
    test_labels_path = f'/content/drive/MyDrive/MLDL_2022/project/pkl_files/D{args.source}_test.pkl'

    with open(test_labels_path, "rb") as f:
        p = pkl.load(f)

    df = pd.DataFrame(p)
    test_labels = df['verb_class'].astype(float)
    test_labels = test_labels.to_numpy()
    test_labels = torch.from_numpy(test_labels).type(torch.LongTensor)
    test_labels.to(device)
    # AVG POOL not needed now since we are trying trm
    # data = torch.mean(train_input_feat, 1)
    
    test_dataset = PersonalizedDataset(test_input_feat, test_labels)
    test_loader = DataLoader(test_dataset, batch_size = 4, shuffle = True)

    # *** TESTIN ***

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            feat, labels = data
            # calculate outputs by running images through the network
            outputs = trm(feat)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(100 * correct // total, '%')
    # print(train_input_ids)

    # out = trm(train_input_feat)
    # print(out[0], out.shape)