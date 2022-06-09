# imports
import torch
import pickle as pkl
import pandas as pd
import numpy as np
from TRNmodule import RelationModuleMultiScaleWithClassifier
from args import parser

if __name__ == "__main__":

    args = parser.parse_args()

    num_frames = 5 # I3D vs TSM
    num_class = 174 # 8????
    img_feature_dim = 512 # 1024
    trm = RelationModuleMultiScaleWithClassifier(img_feature_dim, 5, num_class)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_features_path = f'/content/drive/MyDrive/MLDL_2022/project/EGO_project/Pre-extracted_feat/{args.modality}/ek_{args.backbone}/D{args.source}-D{args.target}_train.pkl'

    # *** TRAIN pkls***

    with open(source_features_path, "rb") as f:
        p = pkl.load(f)

    train_input_feat = p['features']['RGB']
    train_input_ids = p['narration_ids'].split('_')[-1]

    train_labels_path = f'/content/drive/MyDrive/MLDL_2022/project/pkl_files/D{args.source}_train.pkl'

    # TODO: legegre pkls test e fare join per prendere label (sia per train che per test)
    # *** TEST pkls***

    target_features_path = f'/content/drive/MyDrive/MLDL_2022/project/EGO_project/Pre-extracted_feat/{args.modality}/ek_{args.backbone}/D{args.target}-D{args.target}_train.pkl'

    with open(source_features_path, "rb") as f:
        p = pkl.load(f)

    train_input_feat = p['features']['RGB']
    train_input_ids = p['narration_ids'].split('_')[-1]

    train_labels_path = f'/content/drive/MyDrive/MLDL_2022/project/pkl_files/D{args.source}_train.pkl'


    print(train_input_ids)


        #out = trm()