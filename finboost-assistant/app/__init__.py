from flask import Flask
import logging

def create_app():
    app = Flask(__name__)
    
    # Set up logging to file
    logging.basicConfig(filename='error.log', level=logging.ERROR)

    with app.app_context():
        from . import main
        app.register_blueprint(main.bp)
        
    return app
