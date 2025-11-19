
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from clima import obter_dados_clima, criar_dataframes, processar_dados_horarios


class ClothesDataset(Dataset):
    def __init__(self, split, clima_vector):
        dataset = load_dataset("samokosik/clothes_simplified")[split]
        
        self.data = dataset
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        self.clima = torch.tensor(clima_vector, dtype=torch.float32)

        # pegar classes e normalizar rótulos
        labels = [item["label"] for item in dataset]
        classes = sorted(list(set(labels)))
        self.label_map = {old: new for new, old in enumerate(classes)}
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        imagem = self.transform(item["image"])
        label = self.label_map[item["label"]]
        
        clima_feat = self.clima
        
        return imagem, clima_feat, label


def preparar_datasets(local="Curitiba"):
    dados = obter_dados_clima(local)
    df, df_forecast = criar_dataframes(dados)
    clima_vector = processar_dados_horarios(df_forecast)

    train = ClothesDataset("train", clima_vector)
    test = ClothesDataset("test", clima_vector)

    num_classes = len(set([item["label"] for item in load_dataset("samokosik/clothes_simplified")["train"]]))

    return train, test, num_classes
