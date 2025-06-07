import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from config import config

def create_app(config_name='default'):
    """Application factory pattern"""
    
    # Create Flask application
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    # Enable CORS for cross-origin requests
    CORS(app)
    
    # Set up logging
    setup_logging(app)
    
    # Register blueprints
    from app.routes import main_bp, api_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Register error handlers
    register_error_handlers(app)
    
    # Log startup information
    app.logger.info("AIPropertyDetails application started")
    app.logger.info(f"OpenAI Model: {app.config.get('OPENAI_MODEL')}")
    app.logger.info(f"Upload folder: {app.config.get('UPLOAD_FOLDER')}")
    
    return app

def setup_logging(app):
    """Configure application logging"""
    
    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO'))
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    
    # Set Flask app logger level
    app.logger.setLevel(log_level)
    
    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

def register_error_handlers(app):
    """Register error handlers for the application"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Resource not found'}), 404
        return "Page not found", 404
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return jsonify({
            'error': 'File too large',
            'message': f'Maximum file size is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'
        }), 413
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Internal server error: {str(error)}")
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Internal server error'}), 500
        return "Internal server error", 500
    
    @app.errorhandler(Exception)
    def handle_exception(e):
        app.logger.exception("Unhandled exception occurred")
        if request.path.startswith('/api/'):
            return jsonify({'error': 'An unexpected error occurred'}), 500
        return "An unexpected error occurred", 500 