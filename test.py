import torch
import pickle as pkl
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from models import RelationModuleMultiScaleWithClassifier
from models import Baseline_AvgPool_Classifier
from args import parser
from data.dataset import PersonalizedDataset

if __name__ == "__main__":


    def train(train_loader, model, lr, epochs):
      # Define the loss
      loss_criterion = nn.CrossEntropyLoss()
      # Define the optimizer
      optim = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9) #stochastic gradient descent
      total_step = len(train_loader)
      train_loss = []
      for epoch in range(epochs):
        running_loss = 0.0
        for i, batch_images in enumerate(train_loader):
          feat, lbl = batch_images
          # move images to gpu
          feat, lbl = feat.cuda(), lbl.cuda()

          optim.zero_grad() #at each iteration we zero grad the gradient

          out = model(feat)

          loss = loss_criterion(out, lbl)
          # backward the error to the model
          loss.backward()
          optim.step()
          running_loss += loss.item()

          if ((i) % len(train_loader) == 0):
            print(f'Epoch: {epoch+1}\tLoss: {loss.item()}')

        train_loss.append(running_loss/total_step)

      print(f'\ntrain loss: {np.mean(train_loss):.4f}')

    def test(test_loader, model):
      
      correct = 0
      total = 0
      # since we're not training, we don't need to calculate the gradients for our outputs
      with torch.no_grad():
        for data in test_loader:
            feat, labels = data
            # calculate outputs by running images through the network
            outputs = model(feat)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
      print(100 * correct // total, '%')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_features_path = f'/content/drive/MyDrive/MLDL_2022/project/EGO_Project_correct/Pre-extracted_feat/{args.modality}/ek_{args.backbone}/D{args.source}-D{args.target}_train.pkl'
    train_labels_path = f'/content/drive/MyDrive/MLDL_2022/project/pkl_files/D{args.source}_train.pkl'

    target_features_path = f'/content/drive/MyDrive/MLDL_2022/project/EGO_Project_correct/Pre-extracted_feat/{args.modality}/ek_{args.backbone}/D{args.source}-D{args.target}_test.pkl'
    test_labels_path = f'/content/drive/MyDrive/MLDL_2022/project/pkl_files/D{args.target}_test.pkl'

    # *** TRAIN pkls***

    with open(source_features_path, "rb") as f:
        p = pkl.load(f)

    train_input_feat = p['features'][args.modality]
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
      model = RelationModuleMultiScaleWithClassifier(img_feature_dim, num_frames, num_class)
      model.to(device)
    else:
      num_frames = 5
      num_class = 8
      img_feature_dim = 1024
      model = Baseline_AvgPool_Classifier(img_feature_dim, num_class)
      model.to(device)

    train_dataset = PersonalizedDataset(train_input_feat, labels)
    train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)

    # *** TRAINING ***
    train(train_loader = train_loader, model = model, lr = 0.001, epochs = 30)
   

    # *** TEST pkls***


    with open(target_features_path, "rb") as f:
        p = pkl.load(f)

    test_input_feat = p['features'][args.modality]
    test_input_feat= torch.from_numpy(test_input_feat).type(torch.float32)

    with open(test_labels_path, "rb") as f:
        p = pkl.load(f)

    df = pd.DataFrame(p)

    test_labels = df['verb_class'].astype(float)
    test_labels = test_labels.to_numpy()
    test_labels = torch.from_numpy(test_labels).type(torch.LongTensor)

    test_input_feat = test_input_feat.to(device)
    test_labels = test_labels.to(device)
    test_dataset = PersonalizedDataset(test_input_feat, test_labels)
    test_loader = DataLoader(test_dataset, batch_size = 64)

    # *** TESTING ***
    test(test_loader = test_loader, model = model)