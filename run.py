import os
from app import create_app
from config import config

def main():
    """Main application entry point"""
    
    # Get configuration environment
    config_name = os.getenv('FLASK_ENV', 'development')
    
    # Create Flask application
    app = create_app(config_name)
    
    # Get host and port from environment or use defaults
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = config_name == 'development'
    
    print(f"🚀 Starting AIPropertyDetails server...")
    print(f"📍 Environment: {config_name}")
    print(f"🌐 URL: http://{host}:{port}")
    print(f"🔧 Debug mode: {debug}")
    print("="*50)
    
    # Run the application
    app.run(
        host=host,
        port=port,
        debug=debug,
        threaded=True
    )

if __name__ == '__main__':
    main() 