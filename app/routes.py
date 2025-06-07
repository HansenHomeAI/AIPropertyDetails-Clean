import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Blueprint, render_template, request, jsonify, current_app, send_from_directory, flash, redirect, url_for

from app.services.openai_service import OpenAIService
from app.services.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

# Create blueprints
main_bp = Blueprint('main', __name__)
api_bp = Blueprint('api', __name__)

# Initialize services (will be created per request to handle app context)
def get_openai_service():
    return OpenAIService()

def get_document_processor():
    return DocumentProcessor()

# Web interface routes
@main_bp.route('/')
def index():
    """Main landing page"""
    return render_template('index.html')

@main_bp.route('/upload')
def upload_page():
    """File upload page"""
    return render_template('upload.html')

@main_bp.route('/analyze')
def analyze_page():
    """Analysis results page"""
    analysis_id = request.args.get('id')
    if not analysis_id:
        flash('No analysis ID provided', 'error')
        return redirect(url_for('main.index'))
    
    return render_template('analyze.html', analysis_id=analysis_id)

@main_bp.route('/help')
def help_page():
    """Help and documentation page"""
    return render_template('help.html')

# API Routes
@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'model': current_app.config.get('OPENAI_MODEL')
    })

@api_bp.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process a document file"""
    try:
        logger.info("Received file upload request")
        
        # Check if file was provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get document type if provided
        document_type = request.form.get('document_type', 'parcel_map')
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Unsupported file type',
                'allowed_extensions': list(current_app.config['ALLOWED_EXTENSIONS'])
            }), 400
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_extension = Path(filename).suffix
        secure_name = f"{file_id}{file_extension}"
        
        # Save file
        upload_path = current_app.config['UPLOAD_FOLDER'] / secure_name
        file.save(upload_path)
        
        logger.info(f"File saved: {upload_path}")
        
        # Process the file
        processor = get_document_processor()
        file_info = processor.process_uploaded_file(str(upload_path), filename)
        
        if file_info.get('processing_status') == 'error':
            return jsonify({
                'error': 'File processing failed',
                'details': file_info.get('error_message')
            }), 500
        
        # Store file info for analysis
        analysis_data = {
            'file_id': file_id,
            'original_filename': filename,
            'document_type': document_type,
            'upload_timestamp': datetime.utcnow().isoformat(),
            'file_info': file_info,
            'status': 'uploaded'
        }
        
        # In a real application, you'd store this in a database
        # For now, we'll return it directly
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'analysis_data': analysis_data,
            'file_summary': processor.get_file_info_summary(file_info)
        })
        
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        return jsonify({'error': 'Upload failed', 'details': str(e)}), 500

@api_bp.route('/analyze', methods=['POST'])
def analyze_document():
    """Analyze an uploaded document"""
    try:
        data = request.get_json()
        
        if not data or 'file_id' not in data:
            return jsonify({'error': 'file_id is required'}), 400
        
        file_id = data['file_id']
        document_type = data.get('document_type', 'parcel_map')
        
        logger.info(f"Starting analysis for file_id: {file_id}")
        
        # Find the processed image file (from PDF conversion or original image)
        upload_folder = current_app.config['UPLOAD_FOLDER']
        analysis_path = None
        
        # Look for converted PNG file first (from PDF processing)
        png_file = upload_folder / f"{file_id}_page_1.png"
        if png_file.exists():
            analysis_path = str(png_file)
        else:
            # Look for original uploaded file
            for file in upload_folder.glob(f"{file_id}.*"):
                if file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif']:
                    analysis_path = str(file)
                    break
        
        if not analysis_path or not os.path.exists(analysis_path):
            return jsonify({'error': 'Processed image file not found'}), 404
        
        logger.info(f"Using analysis path: {analysis_path}")
        
        # Analyze with OpenAI o4-mini
        openai_service = get_openai_service()
        analysis_result = openai_service.analyze_property_document(analysis_path, document_type)
        
        # Compile final results
        final_result = {
            'analysis_id': str(uuid.uuid4()),
            'file_id': file_id,
            'document_type': document_type,
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'file_processing': {
                'analysis_path': analysis_path,
                'file_type': 'image'
            },
            'ai_analysis': analysis_result,
            'status': 'completed'
        }
        
        logger.info(f"Analysis completed for file_id: {file_id}")
        
        return jsonify({
            'success': True,
            'result': final_result
        })
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return jsonify({'error': 'Analysis failed', 'details': str(e)}), 500

@api_bp.route('/analyze/text', methods=['POST'])
def analyze_text():
    """Analyze text-based legal descriptions directly"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'text field is required'}), 400
        
        text_content = data['text']
        description_type = data.get('type', 'legal_description')
        
        logger.info("Starting text analysis")
        
        # Analyze with OpenAI o4-mini
        openai_service = get_openai_service()
        analysis_result = openai_service.extract_coordinates_from_text(text_content)
        
        # Compile results
        final_result = {
            'analysis_id': str(uuid.uuid4()),
            'text_content': text_content[:500] + '...' if len(text_content) > 500 else text_content,
            'description_type': description_type,
            'analysis_timestamp': datetime.utcnow().isoformat(),
            'ai_analysis': analysis_result,
            'status': 'completed'
        }
        
        return jsonify({
            'success': True,
            'result': final_result
        })
        
    except Exception as e:
        logger.error(f"Text analysis failed: {str(e)}")
        return jsonify({'error': 'Text analysis failed', 'details': str(e)}), 500

@api_bp.route('/validate', methods=['POST'])
def validate_coordinates():
    """Validate extracted coordinates"""
    try:
        data = request.get_json()
        
        if not data or 'coordinates' not in data:
            return jsonify({'error': 'coordinates field is required'}), 400
        
        coordinates = data['coordinates']
        
        # Basic validation
        validation_results = {
            'is_valid': True,
            'issues': [],
            'suggestions': []
        }
        
        # Check coordinate format and ranges
        for i, coord in enumerate(coordinates):
            coord_issues = []
            
            # Check for required fields
            if 'latitude' in coord and 'longitude' in coord:
                lat = coord['latitude']
                lon = coord['longitude']
                
                if lat is not None:
                    if not (-90 <= lat <= 90):
                        coord_issues.append(f"Latitude {lat} out of valid range (-90 to 90)")
                
                if lon is not None:
                    if not (-180 <= lon <= 180):
                        coord_issues.append(f"Longitude {lon} out of valid range (-180 to 180)")
            
            elif 'x_coordinate' in coord and 'y_coordinate' in coord:
                # State plane or UTM coordinates - basic sanity checks
                x = coord['x_coordinate']
                y = coord['y_coordinate']
                
                if x is not None and (x < 0 or x > 10000000):
                    coord_issues.append(f"X coordinate {x} seems outside typical range")
                
                if y is not None and (y < 0 or y > 10000000):
                    coord_issues.append(f"Y coordinate {y} seems outside typical range")
            
            if coord_issues:
                validation_results['issues'].extend([f"Point {i+1}: {issue}" for issue in coord_issues])
                validation_results['is_valid'] = False
        
        # Check for polygon closure
        if len(coordinates) >= 3:
            first_coord = coordinates[0]
            last_coord = coordinates[-1]
            
            if ('latitude' in first_coord and 'latitude' in last_coord):
                if (first_coord.get('latitude') != last_coord.get('latitude') or 
                    first_coord.get('longitude') != last_coord.get('longitude')):
                    validation_results['suggestions'].append("Polygon doesn't appear to close - consider adding the starting point as the last point")
        
        return jsonify({
            'success': True,
            'validation': validation_results
        })
        
    except Exception as e:
        logger.error(f"Coordinate validation failed: {str(e)}")
        return jsonify({'error': 'Validation failed', 'details': str(e)}), 500

@api_bp.route('/export', methods=['POST'])
def export_results():
    """Export analysis results in various formats"""
    try:
        data = request.get_json()
        
        if not data or 'analysis_result' not in data:
            return jsonify({'error': 'analysis_result field is required'}), 400
        
        analysis_result = data['analysis_result']
        export_format = data.get('format', 'json')
        
        if export_format == 'json':
            return jsonify({
                'success': True,
                'export_data': analysis_result,
                'format': 'json'
            })
        
        elif export_format == 'csv':
            # Extract coordinates for CSV export
            coordinates = analysis_result.get('ai_analysis', {}).get('boundary_coordinates', {}).get('vertices', [])
            
            csv_data = []
            csv_data.append('point_id,latitude,longitude,x_coordinate,y_coordinate,description')
            
            for coord in coordinates:
                row = [
                    coord.get('point_id', ''),
                    coord.get('latitude', ''),
                    coord.get('longitude', ''),
                    coord.get('x_coordinate', ''),
                    coord.get('y_coordinate', ''),
                    coord.get('description', '')
                ]
                csv_data.append(','.join(str(x) for x in row))
            
            return jsonify({
                'success': True,
                'export_data': '\n'.join(csv_data),
                'format': 'csv'
            })
        
        elif export_format == 'kml':
            # Generate KML format for Google Earth
            coordinates = analysis_result.get('ai_analysis', {}).get('boundary_coordinates', {}).get('vertices', [])
            
            kml_coordinates = []
            for coord in coordinates:
                if coord.get('longitude') and coord.get('latitude'):
                    kml_coordinates.append(f"{coord['longitude']},{coord['latitude']},0")
            
            kml_data = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Property Boundary</name>
    <Placemark>
      <name>Property Parcel</name>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
              {' '.join(kml_coordinates)}
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>"""
            
            return jsonify({
                'success': True,
                'export_data': kml_data,
                'format': 'kml'
            })
        
        else:
            return jsonify({'error': f'Unsupported export format: {export_format}'}), 400
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        return jsonify({'error': 'Export failed', 'details': str(e)}), 500

# Utility functions
def allowed_file(filename):
    """Check if filename has an allowed extension"""
    if not filename:
        return False
    
    file_ext = Path(filename).suffix.lower()
    return file_ext.lstrip('.') in current_app.config['ALLOWED_EXTENSIONS']

# Error handlers for blueprints
@main_bp.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404

@main_bp.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error"), 500 