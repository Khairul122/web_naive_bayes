from flask import Flask
from .AuthRoute import auth_bp
from .DashboardRoute import dashboard_bp
from app.routes.ScrappingRoute import scrapping_bp
from app.routes.PreprocessingRoute import preprocessing_bp
from app.routes.SentimenRoute import sentimen_bp
from app.routes.KonversiRoute import konversi_bp
from app.routes.NBCRoute import nbc_bp


def register_routes(app: Flask):
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(scrapping_bp)
    app.register_blueprint(preprocessing_bp)
    app.register_blueprint(sentimen_bp)
    app.register_blueprint(konversi_bp)
    app.register_blueprint(nbc_bp)


