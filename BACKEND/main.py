from flask import Flask
from flask_cors import CORS 
import os
from controllers.recommendations import recommendations_bp
from controllers.weather import weather_bp
from controllers.clothes import clothes_bp 

app = Flask(__name__)
CORS(app)

app.register_blueprint(recommendations_bp, url_prefix='/recomendar')
app.register_blueprint(weather_bp, url_prefix='/api/clima')
app.register_blueprint(clothes_bp, url_prefix='/api/clothes')


if __name__ == '__main__':
    print("\nServidor Backend rodando!")
    print("Acesse o Frontend em: index.html")
    print("API disponível em: http://localhost:5000/\n")
    app.run(debug=True, host='0.0.0.0', port=5000)