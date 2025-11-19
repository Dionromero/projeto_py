
import torch
import os
from treinamento import treinar
from clima import processar_dados_horarios
from dados import preparar_datasets
from imagem import predizer_imagem
#from predict import predizer_imagem

1
MODELO_PATH = "modelo_final.pth"


def salvar_modelo(modelo):
    torch.save(modelo.state_dict(), MODELO_PATH)
    print(f"\n💾 Modelo salvo em: {MODELO_PATH}")


def carregar_modelo():
    if not os.path.exists(MODELO_PATH):
        print("❌ Nenhum modelo salvo encontrado. Treine primeiro.")
        return None

    print("📦 Carregando modelo salvo...")
    train_ds, _, num_classes = preparar_datasets("Curitiba")
    clima_dim = len(train_ds.clima)

    modelo = processar_dados_horarios (num_classes, clima_dim)
    modelo.load_state_dict(torch.load(MODELO_PATH, map_location="cpu"))
    modelo.eval()

    print("✅ Modelo carregado com sucesso!")
    return modelo


def menu():
    while True:
        print("\n==============================")
        print("        MENU PRINCIPAL")
        print("==============================")
        print("1 → Treinar modelo")
        print("2 → Carregar modelo salvo")
        print("3 → Predizer imagem nova")
        print("4 → Sair")
        print("==============================")

        opc = input("Escolha uma opção: ")

        # -------------------------------
        if opc == "1":
            print("\n🚀 Iniciando treinamento...")
            modelo = treinar(epochs=10)
            salvar_modelo(modelo)

        # -------------------------------
        elif opc == "2":
            modelo = carregar_modelo()

        # -------------------------------
        elif opc == "3":
            modelo = carregar_modelo()
            if modelo is None:
                continue

            caminho = input("\nCaminho da imagem: ")
            
            if not os.path.exists(caminho):
                print("❌ Caminho inválido!")
                continue
            
            predizer_imagem(caminho, MODELO_PATH)

        # -------------------------------
        elif opc == "4":
            print("\n👋 Encerrando programa.")
            break

        else:
            print("❌ Opção inválida!")


if __name__ == "__main__":
    menu()
