import os
from datasets import load_dataset
from PIL import Image

def salvar_exemplos_frontend(dataset_name="samokosik/clothes_simplified", split="train", target_dir="FRONTEND/images"):
    
    print("Iniciando a automação: Salvando exemplos de imagens para o Frontend...")

    try:
        # Tenta carregar o dataset
        dataset = load_dataset(dataset_name)[split]
    except Exception as e:
        print(f" Erro ao carregar o dataset {dataset_name}: {e}")
        return

    # Mapeamento de Labels para garantir que correspondem ao backend
    try:
        labels_originais = sorted(list(set([item["label"] for item in dataset])))
        label_map = {old: new for new, old in enumerate(labels_originais)}
    except KeyError:
        print(" Erro: O dataset não possui a chave 'label'.")
        return

    exemplos_encontrados = {} # {classe_mapeada: 'image_id'}

    # Loop com enumerate para gerar o ID igual ao do backend
    for idx, item in enumerate(dataset):
        original_label = item["label"]
        mapeada_label = label_map.get(original_label)
        
        #  GERA O ID BASEADO NO ÍNDICE (00000000, 00000001...)
        image_id = f"{idx:08d}" 
        img_pil = item["image"] 

        if mapeada_label is None or mapeada_label in exemplos_encontrados:
            continue

        # Define os caminhos
        class_folder = str(mapeada_label)
        caminho_pasta = os.path.join(target_dir, class_folder)
        nome_arquivo = f"image_{image_id}.jpg" 
        caminho_completo = os.path.join(caminho_pasta, nome_arquivo)

        # Cria a pasta se não existir
        os.makedirs(caminho_pasta, exist_ok=True)

        # Salva a imagem
        try:
            img_pil.save(caminho_completo, 'JPEG')
            print(f" Salvo: Classe {mapeada_label} -> {caminho_completo}")
            exemplos_encontrados[mapeada_label] = image_id 
        except Exception as e:
            print(f" Erro ao salvar imagem {image_id}: {e}")

        # Pára quando tiver 1 exemplo de cada classe
        if len(exemplos_encontrados) >= len(labels_originais):
            break

    print(f"\n Processo concluído! {len(exemplos_encontrados)} imagens prontas para o site.")

if __name__ == "__main__":
    salvar_exemplos_frontend()