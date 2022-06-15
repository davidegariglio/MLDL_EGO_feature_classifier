from typing_extensions import ParamSpecArgs
import torch
import pickle as pkl
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from models import RelationModuleMultiScaleWithClassifier
from args import parser
from data.dataset import PersonalizedDataset

if __name__ == "__main__":


    def train():
      # TODO
      pass

    def test():
      # TODO
      pass
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_features_path = f'/content/drive/MyDrive/MLDL_2022/project/EGO_Project_correct/Pre-extracted_feat/{args.modality}/ek_{args.backbone}/D{args.source}-D{args.target}_train.pkl'
    train_labels_path = f'/content/drive/MyDrive/MLDL_2022/project/pkl_files/D{args.source}_train.pkl'

    target_features_path = f'/content/drive/MyDrive/MLDL_2022/project/EGO_project/Pre-extracted_feat/{args.modality}/ek_{args.backbone}/D{args.source}-D{args.target}_test.pkl'
    test_labels_path = f'/content/drive/MyDrive/MLDL_2022/project/pkl_files/D{args.target}_test.pkl'

    # *** TRAIN pkls***

    with open(source_features_path, "rb") as f:
        p = pkl.load(f)

    train_input_feat = p['features']['RGB']
    train_input_feat= torch.from_numpy(train_input_feat).type(torch.float32)
    train_input_feat = train_input_feat.to(device)


    with open(train_labels_path, "rb") as f:
        p = pkl.load(f)

    df = pd.DataFrame(p)
    labels = df['verb_class'].astype(float)
    labels = labels.to_numpy()
    labels = torch.from_numpy(labels).type(torch.LongTensor)
    labels = labels.to(device)

    if args.backbone == 'tsm':
      num_frames = 5 
      num_class = 8
      img_feature_dim = 2048
      trm = RelationModuleMultiScaleWithClassifier(img_feature_dim, num_frames, num_class)
      trm.to(device)
    else:
      num_frames = 5
      num_class = 8
      img_feature_dim = 1024
      data = torch.mean(train_input_feat, 1)
    
    train_dataset = PersonalizedDataset(train_input_feat, labels)
    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)

    # *** TRAINING ***

    # Define the loss
    loss_criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    lr = 0.001
    optim = torch.optim.SGD(trm.parameters(), lr = lr, momentum = 0.9) #stochastic gradient descent
    epochs = 30
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


    with open(target_features_path, "rb") as f:
        p = pkl.load(f)

    test_input_feat = p['features']['RGB']
    test_input_feat= torch.from_numpy(test_input_feat).type(torch.float32)

    with open(test_labels_path, "rb") as f:
        p = pkl.load(f)

    df = pd.DataFrame(p)
    test_labels = df['verb_class'].astype(float)
    test_labels = test_labels.to_numpy()
    test_labels = torch.from_numpy(test_labels).type(torch.LongTensor)
    # AVG POOL not needed now since we are trying trm
    # data = torch.mean(train_input_feat, 1)
    test_input_feat = test_input_feat.to(device)
    test_labels = test_labels.to(device)
    test_dataset = PersonalizedDataset(test_input_feat, test_labels)
    test_loader = DataLoader(test_dataset, batch_size = 64)

    # *** TESTING ***

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