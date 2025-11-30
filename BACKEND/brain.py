import torch
import torch.nn as nn

ROUPAS_CIMA = [
    "Camiseta Básica", "Camiseta Oversized", "Camiseta Dry", "Regata Sport", 
    "Camisa Social", "Suéter", "Top Cropped", "Blusa de Seda", "Camisa Polo",
    "Camisa Jeans", "Bata Estampada", "Camisa Xadrez", "Top de Brilho"
]

ROUPAS_BAIXO = [
    "Jeans Reto", "Jeans Rasgado", "Calça Cargo", "Calça Social", "Legging", 
    "Short Esportivo", "Saia Mídi", "Saia Plissada", "Calça Jogger", 
    "Short Jeans", "Calça de Couro", "Calça Wide Leg", "Saia Longa", 
    "Minissaia", "Bermuda", "Calça Chino", "Calça Grossa"
]

ROUPAS_CASACO = [
    "Nada", "Moletom Hoodie", "Jaqueta Jeans", "Blazer", "Jaqueta Sport", 
    "Sobretudo", "Cardigan", "Jaqueta Bomber", "Jaqueta de Couro", 
    "Jaqueta Curta", "Jaqueta Utilitária", "Jaqueta de Pelo", "Trench Coat"
]

ESTILOS = ["Casual", "Formal", "Esportivo", "Inverno", "Verao", "Streetwear", "Minimalista", "Vintage", "Y2K", "Boho", "Grunge", "Preppy", "Glam", "Workwear"]
GENEROS = ["male", "female", "unisex"]

class NeuralStylist(nn.Module):
    def __init__(self):
        super(NeuralStylist, self).__init__()
        input_size = 1 + len(GENEROS) + len(ESTILOS)
        
        self.shared = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.head_cima = nn.Linear(64, len(ROUPAS_CIMA))
        self.head_baixo = nn.Linear(64, len(ROUPAS_BAIXO))
        self.head_casaco = nn.Linear(64, len(ROUPAS_CASACO))
        
    def forward(self, x):
        x = self.shared(x)
        return self.head_cima(x), self.head_baixo(x), self.head_casaco(x)

def preparar_dados_entrada(temp, genero, estilo):
    t_norm = (float(temp) + 10) / 50.0
    g_vec = [0] * len(GENEROS)
    if genero in GENEROS: g_vec[GENEROS.index(genero)] = 1
    s_vec = [0] * len(ESTILOS)
    if estilo in ESTILOS: s_vec[ESTILOS.index(estilo)] = 1
    
    entrada = [t_norm] + g_vec + s_vec
    return torch.tensor([entrada], dtype=torch.float32)