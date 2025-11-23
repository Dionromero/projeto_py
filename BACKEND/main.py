from flask import Flask
from flask_cors import CORS 
import os

# Imports dos Controllers (Blueprints)
# Certifique-se de que os arquivos existem nas pastas corretas
from controllers.recommendations import recommendations_bp
from controllers.weather import weather_bp
from controllers.clothes import clothes_bp 

# from infra.db import db # (Opcional: Descomente se já tiver o banco configurado)

app = Flask(__name__)

# 🔥 Habilita CORS para todas as rotas
# Isso permite que seu index.html e cadastro.html acessem o Python
CORS(app)

# --- Configuração do Banco (Opcional por enquanto) ---
# app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///app.db')
# db.init_app(app)

# --- Registro das Rotas ---

# 1. Rota de Recomendação
# O Frontend chama: http://localhost:5000/recomendar
# Se no recommendations.py a rota for '/', aqui o prefixo deve ser '/recomendar'
app.register_blueprint(recommendations_bp, url_prefix='/recomendar')

# 2. Rota de Clima
# O Frontend chama: http://localhost:5000/api/clima
# Se no weather.py a rota for '/', aqui o prefixo deve ser '/api/clima'
app.register_blueprint(weather_bp, url_prefix='/api/clima')

# 3. Rota de Cadastro de Roupas
# O Frontend chama: POST http://localhost:5000/api/clothes
app.register_blueprint(clothes_bp, url_prefix='/api/clothes')


if __name__ == '__main__':
    print("\n🚀 Servidor Backend rodando!")
    print("Acesse o Frontend em: index.html")
    print("API disponível em: http://localhost:5000/\n")
    app.run(debug=True, host='0.0.0.0', port=5000)