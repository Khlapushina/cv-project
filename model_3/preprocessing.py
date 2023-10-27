
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import datasets
from torchvision import transforms as T
import torchutils as tu
from typing import Tuple
from tqdm import tqdm
from torchvision.io import read_image


preprocessing = T.Compose(
    [
        #T.ToPILImage(),
        #T.Resize((400, 400)), 
        T.ToTensor()
    ]
)

def preprocess(img):
    return preprocessing(img)