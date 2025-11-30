import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from brain import NeuralStylist, preparar_dados_entrada, ROUPAS_CIMA, ROUPAS_BAIXO, ROUPAS_CASACO, GENEROS, ESTILOS
from motor_recomendacao import filtrar

def treinar_agora():
    print("🧠 [Deep Learning] Iniciando treinamento da IA...")
    
    modelo = NeuralStylist()
    optimizer = optim.Adam(modelo.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    
    X_train = []
    y_cima = []
    y_baixo = []
    y_casaco = []
    
    print(" Gerando dados de treino...")
    for _ in range(20000):
        temp = random.randint(-5, 35)
        gen = random.choice(GENEROS)
        est = random.choice(ESTILOS)
        
        gabarito, _ = filtrar(None, temp, gen, est)
        
        # Validação se existe peça
        if not gabarito['cima'] and not gabarito['corpo_inteiro']: continue
        if not gabarito['baixo'] and not gabarito['corpo_inteiro']: continue
            
        # Escolhe peças para ensinar a IA
        if gabarito['corpo_inteiro'] and random.random() > 0.5:
             peca_c = random.choice(gabarito['corpo_inteiro'])
             peca_b = "Jeans Reto" 
        else:
             if not gabarito['cima'] or not gabarito['baixo']: continue
             peca_c = random.choice(gabarito['cima'])
             peca_b = random.choice(gabarito['baixo'])

        peca_k = random.choice(gabarito['casaco']) if gabarito['casaco'] else "Nada"
        
        if peca_c in ROUPAS_CIMA and peca_b in ROUPAS_BAIXO and peca_k in ROUPAS_CASACO:
            tensor_in = preparar_dados_entrada(temp, gen, est)
            
            X_train.append(tensor_in)
            y_cima.append(torch.tensor([ROUPAS_CIMA.index(peca_c)]))
            y_baixo.append(torch.tensor([ROUPAS_BAIXO.index(peca_b)]))
            y_casaco.append(torch.tensor([ROUPAS_CASACO.index(peca_k)]))

    if not X_train:
        print(" Erro: Não foi possível gerar dados. Verifique a compatibilidade entre motor_recomendacao.py e brain.py")
        return

    X_batch = torch.cat(X_train)
    Y_c_batch = torch.cat(y_cima)
    Y_b_batch = torch.cat(y_baixo)
    Y_k_batch = torch.cat(y_casaco)
    
    modelo.train()
    print(" Treinando redes neurais...")
    for epoch in range(150):
        optimizer.zero_grad()
        out_c, out_b, out_k = modelo(X_batch)
        loss = loss_fn(out_c, Y_c_batch) + loss_fn(out_b, Y_b_batch) + loss_fn(out_k, Y_k_batch)
        loss.backward()
        optimizer.step()
        
    torch.save(modelo.state_dict(), "cerebro_estilista.pth")
    print(" Cérebro IA treinado e salvo!")

if __name__ == "__main__":
    treinar_agora()