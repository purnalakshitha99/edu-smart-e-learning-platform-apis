
import os
from dotenv import load_dotenv
from flask import Flask
from static.logger import logging
from controllers.auth_controller import auth_controller
from controllers.ethical_benchmark_controller import ethical_benchmark_controller
from controllers.test_controller import test_controller
from flask_jwt_extended import JWTManager
from db import mongo
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

CORS(app, origins="http://localhost:5173")


app.config['MONGO_URI'] = os.getenv("MONGO_URI")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")

print("MONGO_URI is:", app.config["MONGO_URI"])


#mongo app start
mongo.init_app(app)

logging.info(f'Preprocessed Text : {"Flask Server is started"}')

jwt = JWTManager(app)

# ROUTES Declaration
app.register_blueprint(auth_controller, url_prefix='/auth')
app.register_blueprint(ethical_benchmark_controller, url_prefix='/ethical_benchmark')
app.register_blueprint(test_controller, url_prefix='/test')

if __name__ == "__main__":
    app.run(debug=True) 