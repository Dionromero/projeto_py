
from flask import Blueprint, request, jsonify
import sys
import os
# Adiciona o diretório pai ao caminho do sistema para encontrar 'clima_api' e 'motor_recomendacao'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

clothes_bp = Blueprint('clothes', __name__)

roupas_db = [] 

@clothes_bp.route('/', methods=['POST'])
def create_cloth():
    data = request.json
    
    nova_roupa = {
        "id": len(roupas_db) + 1,
        "name": data.get('name'),
        "image_path": data.get('image_path'),
        "tags": data.get('tags', [])
    }
    
    roupas_db.append(nova_roupa)
    
    print(f" Nova roupa cadastrada: {nova_roupa['name']} | Tags: {nova_roupa['tags']}")
    
    return jsonify({"msg": "Roupa cadastrada com sucesso!", "dados": nova_roupa}), 201

@clothes_bp.route('/', methods=['GET'])
def list_clothes():
    return jsonify(roupas_db)