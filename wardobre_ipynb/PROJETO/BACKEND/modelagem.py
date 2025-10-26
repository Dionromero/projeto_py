import clima 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from datasets import load_dataset


# Carregar o dataset de roupas
dataset = load_dataset("samokosik/clothes_simplified")
df_train = dataset['train']
df_test = dataset['test']

# Função para preparar DataLoader
