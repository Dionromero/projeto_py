import sys
import os
from flask import Flask
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from controllers.recommendations import recommendations_bp
from controllers.weather import weather_bp
from controllers.clothes import clothes_bp

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.register_blueprint(recommendations_bp, url_prefix='/recomendar')
app.register_blueprint(weather_bp, url_prefix='/api/clima')
app.register_blueprint(clothes_bp, url_prefix='/api/clothes')

@app.route('/')
def home():
    return "<h1>Servidor Neural Online </h1>"

if __name__ == '__main__':
    print("\n" + "="*40)
    print(" SERVIDOR INICIADO!")
    print(" API: http://127.0.0.1:5000")
    print("="*40 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)