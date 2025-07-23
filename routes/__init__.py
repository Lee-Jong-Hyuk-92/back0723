from .application_routes import application_bp

def register_routes(app):
    app.register_blueprint(application_bp, url_prefix='/api')