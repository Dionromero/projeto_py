import sys
import os
from flask import Blueprint, request, jsonify

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

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
    return jsonify({"msg": "Roupa cadastrada com sucesso!", "dados": nova_roupa}), 201

@clothes_bp.route('/', methods=['GET'])
def list_clothes():
    return jsonify(roupas_db)