
import torch
import os
from treinamento import treinar
from imagem import predizer_imagem
from modelo import ModeloVisaoClimaLite
from dados import preparar_datasets

MODELO_PATH = "modelo_final_b.pth"


# --------------------------------------------------
# Salvar e carregar modelo
# --------------------------------------------------

def salvar_modelo(modelo, path=MODELO_PATH):
    torch.save(modelo.state_dict(), path)
    print(f"\n Modelo salvo em: {path}")


def carregar_modelo():
    if not os.path.exists(MODELO_PATH):
        print("\n Nenhum modelo salvo encontrado! Treine o modelo antes.")
        return None

    print("\n Carregando modelo salvo...")

    # pegar dados só para saber num_classes e clima_dim
    train_ds, _, num_classes = preparar_datasets("Curitiba")
    clima_dim = len(train_ds.clima)

    modelo = ModeloVisaoClimaLite(num_classes, clima_dim, freeze_backbone=True)
    modelo.load_state_dict(torch.load(MODELO_PATH, map_location="cpu"))
    modelo.eval()

    print(" Modelo carregado com sucesso!")
    return modelo


# --------------------------------------------------
# Listar imagens em uma pasta
# --------------------------------------------------

def listar_imagens(pasta="imagens"):
    print("\n Imagens disponíveis:")

    if not os.path.exists(pasta):
        print(f" Pasta '{pasta}' não existe. Crie e coloque imagens nela.")
        return []

    arquivos = [f for f in os.listdir(pasta) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    if not arquivos:
        print(" Nenhuma imagem encontrada.")
        return []

    for i, nome in enumerate(arquivos, 1):
        print(f" {i}. {os.path.join(pasta, nome)}")

    return arquivos


# --------------------------------------------------
# Menu interativo
# --------------------------------------------------

def menu():
    while True:
        print("\n==============================")
        print("        MENU PRINCIPAL")
        print("==============================")
        print("1 → Treinar modelo (rápido — backbone congelado)")
        print("2 → Fine-tune (aprender melhor — mais lento)")
        print("3 → Carregar modelo salvo")
        print("4 → Predizer imagem")
        print("5 → Sair")
        print("==============================")

        opc = input("Escolha uma opção: ").strip()

        # -----------------------------------------------------
        # Treino rápido (forma B — backbone congelado)
        # -----------------------------------------------------
        if opc == "1":
            print("\n Iniciando treinamento rápido (backbone congelado)...")
            modelo = treinar(epochs=3, batch_size=128, lr=1e-3, freeze_backbone=True)
            salvar_modelo(modelo)

        # -----------------------------------------------------
        # Fine-tune (descongela backbone)
        # -----------------------------------------------------
        elif opc == "2":
            print("\n Fine-tune iniciado (descongelando partes do backbone)...")
            modelo = treinar(epochs=5, batch_size=64, lr=5e-4, freeze_backbone=False)
            salvar_modelo(modelo)

        # -----------------------------------------------------
        # Carregar modelo salvo
        # -----------------------------------------------------
        elif opc == "3":
            carregar_modelo()

        # -----------------------------------------------------
        # Predição
        # -----------------------------------------------------
        elif opc == "4":
            modelo = carregar_modelo()
            if modelo is None:
                continue

            imagens = listar_imagens()

            print("\n Digite o caminho OU selecione um número:")
            caminho = input("→ ").strip()

            # se for número
            if caminho.isdigit():
                idx = int(caminho) - 1
                if 0 <= idx < len(imagens):
                    caminho = os.path.join("imagens", imagens[idx])
                else:
                    print(" Número inválido!")
                    continue

            # validar caminho
            if not os.path.exists(caminho):
                print(" Caminho inválido!")
                continue

            classe = predizer_imagem(caminho, MODELO_PATH)
            print(f"\n Classe prevista: {classe}")

        # -----------------------------------------------------
        # Sair
        # -----------------------------------------------------
        elif opc == "5":
            print("\n Encerrando programa.")
            break

        else:
            print(" Opção inválida!")


if __name__ == "__main__":
    menu()
