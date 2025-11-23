from flask import Blueprint, request, jsonify

clothes_bp = Blueprint('clothes', __name__)

# Simulação de banco de dados em memória (para teste)
# Quando você reiniciar o servidor, isso apaga. 
# Futuramente, trocaremos isso pelo SQL/Banco de dados.
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
    
    print(f"👕 Nova roupa cadastrada: {nova_roupa['name']} | Tags: {nova_roupa['tags']}")
    
    return jsonify({"msg": "Roupa cadastrada com sucesso!", "dados": nova_roupa}), 201

@clothes_bp.route('/', methods=['GET'])
def list_clothes():
    return jsonify(roupas_db)