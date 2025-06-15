from flask import Flask
from app.extension import db
from app.routes import register_routes


def create_app():
    app = Flask(__name__, template_folder='templates')
    app.secret_key = 'supersecretkey'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/db_naive_bayes'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    register_routes(app)

    return app
