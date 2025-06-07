import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    
    @property
    def validated_openai_key(self):
        """Get OpenAI API key with validation"""
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return self.OPENAI_API_KEY
    OPENAI_MODEL = 'o4-mini-2025-04-16'  # Using the latest o4-mini model
    OPENAI_MAX_TOKENS = 4000
    OPENAI_TEMPERATURE = 0.1  # Lower temperature for more consistent analysis
    
    # Application Configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = Path('uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'tiff', 'bmp'}
    
    # Analysis Configuration
    COORDINATE_PRECISION = 6  # Decimal places for coordinates
    MAX_PROCESSING_TIME = 300  # 5 minutes max processing time
    
    # Create directories if they don't exist
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    Path('static').mkdir(exist_ok=True)
    Path('templates').mkdir(exist_ok=True)
    
    @staticmethod
    def init_app(app):
        """Initialize application with this config"""
        pass


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False
    
    # More verbose logging in development
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Production security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # Production logging
    LOG_LEVEL = 'INFO'
    
    @classmethod
    def init_app(cls, app):
        Config.init_app(app)
        
        # Log to syslog in production
        import logging
        from logging.handlers import SysLogHandler
        syslog_handler = SysLogHandler()
        syslog_handler.setLevel(logging.WARNING)
        app.logger.addHandler(syslog_handler)


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    WTF_CSRF_ENABLED = False
    
    # Use temporary directory for testing
    UPLOAD_FOLDER = Path('/tmp/test_uploads')
    UPLOAD_FOLDER.mkdir(exist_ok=True)


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 