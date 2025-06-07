import os
import logging
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import cv2
import numpy as np
from pdf2image import convert_from_path
from flask import current_app

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Service for processing and preparing documents for analysis"""
    
    def __init__(self):
        self.supported_formats = {
            'image': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif'],
            'pdf': ['.pdf'],
            'text': ['.txt']
        }
    
    def process_uploaded_file(self, file_path: str, original_filename: str) -> Dict:
        """
        Process an uploaded file and prepare it for analysis
        
        Args:
            file_path: Path to the uploaded file
            original_filename: Original name of the uploaded file
            
        Returns:
            Dictionary with processing results and file information
        """
        try:
            logger.info(f"Processing uploaded file: {original_filename}")
            
            # Determine file type
            file_info = self._analyze_file(file_path, original_filename)
            
            if file_info['file_type'] == 'image':
                processed_info = self._process_image_file(file_path, file_info)
            elif file_info['file_type'] == 'pdf':
                processed_info = self._process_pdf_file(file_path, file_info)
            elif file_info['file_type'] == 'text':
                processed_info = self._process_text_file(file_path, file_info)
            else:
                raise ValueError(f"Unsupported file type: {file_info['file_type']}")
            
            # Add processing metadata
            processed_info.update({
                'original_filename': original_filename,
                'processed_path': file_path,
                'processing_status': 'success'
            })
            
            logger.info(f"Successfully processed file: {original_filename}")
            return processed_info
            
        except Exception as e:
            logger.error(f"Failed to process file {original_filename}: {str(e)}")
            return {
                'original_filename': original_filename,
                'processing_status': 'error',
                'error_message': str(e)
            }
    
    def _analyze_file(self, file_path: str, filename: str) -> Dict:
        """Analyze file and determine its type and properties"""
        file_ext = Path(filename).suffix.lower()
        file_size = os.path.getsize(file_path)
        
        # Determine file type category
        file_type = None
        for category, extensions in self.supported_formats.items():
            if file_ext in extensions:
                file_type = category
                break
        
        if not file_type:
            raise ValueError(f"Unsupported file extension: {file_ext}")
        
        # Get MIME type
        mime_type, _ = mimetypes.guess_type(filename)
        
        return {
            'file_type': file_type,
            'file_extension': file_ext,
            'file_size': file_size,
            'mime_type': mime_type,
            'is_supported': True
        }
    
    def _process_image_file(self, file_path: str, file_info: Dict) -> Dict:
        """Process image files and extract metadata"""
        try:
            # Load image with PIL
            image = Image.open(file_path)
            
            # Get image properties
            width, height = image.size
            mode = image.mode
            format_name = image.format
            
            # Check if image needs optimization
            optimized_path = None
            if self._should_optimize_image(image, file_info['file_size']):
                optimized_path = self._optimize_image(file_path, image)
            
            # Analyze image quality
            quality_assessment = self._assess_image_quality(file_path)
            
            return {
                **file_info,
                'image_properties': {
                    'width': width,
                    'height': height,
                    'mode': mode,
                    'format': format_name,
                    'has_transparency': mode in ('RGBA', 'LA') or 'transparency' in image.info
                },
                'optimized_path': optimized_path,
                'analysis_path': optimized_path or file_path,
                'quality_assessment': quality_assessment,
                'ready_for_analysis': True
            }
            
        except Exception as e:
            logger.error(f"Failed to process image file: {str(e)}")
            raise
    
    def _process_pdf_file(self, file_path: str, file_info: Dict) -> Dict:
        """Process PDF files and convert to images"""
        try:
            logger.info("Converting PDF to images for analysis")
            
            # Convert PDF to images
            images = convert_from_path(file_path, dpi=300, fmt='png')
            
            # Save converted images
            image_paths = []
            upload_dir = Path(file_path).parent
            
            for i, image in enumerate(images):
                image_filename = f"{Path(file_path).stem}_page_{i+1}.png"
                image_path = upload_dir / image_filename
                image.save(image_path, 'PNG')
                image_paths.append(str(image_path))
            
            # Analyze each page for quality
            page_assessments = []
            for image_path in image_paths:
                quality = self._assess_image_quality(image_path)
                page_assessments.append(quality)
            
            return {
                **file_info,
                'pdf_properties': {
                    'num_pages': len(images),
                    'converted_images': image_paths,
                    'page_assessments': page_assessments
                },
                'analysis_path': image_paths[0] if image_paths else None,  # Use first page by default
                'ready_for_analysis': len(image_paths) > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to process PDF file: {str(e)}")
            raise
    
    def _process_text_file(self, file_path: str, file_info: Dict) -> Dict:
        """Process text files containing legal descriptions"""
        try:
            # Read text content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic text analysis
            word_count = len(content.split())
            line_count = len(content.splitlines())
            char_count = len(content)
            
            # Check for coordinate-like patterns
            coordinate_patterns = self._detect_coordinate_patterns(content)
            
            return {
                **file_info,
                'text_properties': {
                    'content': content,
                    'word_count': word_count,
                    'line_count': line_count,
                    'char_count': char_count,
                    'coordinate_patterns': coordinate_patterns
                },
                'analysis_content': content,
                'ready_for_analysis': True
            }
            
        except Exception as e:
            logger.error(f"Failed to process text file: {str(e)}")
            raise
    
    def _should_optimize_image(self, image: Image.Image, file_size: int) -> bool:
        """Determine if image should be optimized for better analysis"""
        
        # Optimize if file is too large (> 10MB)
        if file_size > 10 * 1024 * 1024:
            return True
        
        # Optimize if image is too large in dimensions
        width, height = image.size
        if width > 4000 or height > 4000:
            return True
        
        # Optimize if image has too many channels or is not in a good format for analysis
        if image.mode not in ('RGB', 'L', 'RGBA'):
            return True
        
        return False
    
    def _optimize_image(self, file_path: str, image: Image.Image) -> str:
        """Optimize image for better analysis performance"""
        try:
            # Create optimized filename
            optimized_path = file_path.replace('.', '_optimized.')
            if not optimized_path.endswith(('.png', '.jpg', '.jpeg')):
                optimized_path += '.png'
            
            # Convert to RGB if necessary
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            # Resize if too large
            width, height = image.size
            max_dimension = 3000
            
            if width > max_dimension or height > max_dimension:
                ratio = min(max_dimension / width, max_dimension / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Save optimized image
            if optimized_path.endswith('.png'):
                image.save(optimized_path, 'PNG', optimize=True)
            else:
                image.save(optimized_path, 'JPEG', optimize=True, quality=95)
            
            logger.info(f"Image optimized and saved to: {optimized_path}")
            return optimized_path
            
        except Exception as e:
            logger.error(f"Failed to optimize image: {str(e)}")
            return file_path  # Return original if optimization fails
    
    def _assess_image_quality(self, image_path: str) -> Dict:
        """Assess image quality for analysis purposes"""
        try:
            # Load image with OpenCV for quality analysis
            img = cv2.imread(image_path)
            
            if img is None:
                return {'quality_score': 0, 'issues': ['Could not load image']}
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness and contrast
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Assess quality factors
            issues = []
            quality_factors = []
            
            # Sharpness assessment
            if laplacian_var < 100:
                issues.append('Image appears blurry')
                sharpness_score = 0.3
            elif laplacian_var < 500:
                sharpness_score = 0.7
            else:
                sharpness_score = 1.0
            quality_factors.append(sharpness_score)
            
            # Brightness assessment
            if brightness < 50:
                issues.append('Image is too dark')
                brightness_score = 0.5
            elif brightness > 200:
                issues.append('Image is too bright')
                brightness_score = 0.7
            else:
                brightness_score = 1.0
            quality_factors.append(brightness_score)
            
            # Contrast assessment
            if contrast < 30:
                issues.append('Low contrast')
                contrast_score = 0.5
            else:
                contrast_score = 1.0
            quality_factors.append(contrast_score)
            
            # Calculate overall quality score
            quality_score = np.mean(quality_factors)
            
            return {
                'quality_score': round(quality_score, 2),
                'sharpness': round(laplacian_var, 2),
                'brightness': round(brightness, 2),
                'contrast': round(contrast, 2),
                'issues': issues,
                'suitable_for_analysis': quality_score > 0.5
            }
            
        except Exception as e:
            logger.error(f"Failed to assess image quality: {str(e)}")
            return {'quality_score': 0.5, 'issues': ['Quality assessment failed']}
    
    def _detect_coordinate_patterns(self, text: str) -> List[str]:
        """Detect potential coordinate patterns in text"""
        import re
        
        patterns = []
        
        # Latitude/Longitude patterns
        lat_long_pattern = r'[-+]?\d{1,3}\.\d+[°]?\s*[NS]?,?\s*[-+]?\d{1,3}\.\d+[°]?\s*[EW]?'
        matches = re.findall(lat_long_pattern, text, re.IGNORECASE)
        patterns.extend([f"lat_long: {match}" for match in matches])
        
        # UTM/State Plane patterns
        utm_pattern = r'\d{6,7}\.\d+[mMfF]?\s*[NS],?\s*\d{6,7}\.\d+[mMfF]?\s*[EW]'
        matches = re.findall(utm_pattern, text)
        patterns.extend([f"utm_coords: {match}" for match in matches])
        
        # Bearing patterns
        bearing_pattern = r'[NS]\s*\d{1,3}[°]\s*\d{1,2}[\'′]\s*\d{1,2}[\"″]?\s*[EW]'
        matches = re.findall(bearing_pattern, text, re.IGNORECASE)
        patterns.extend([f"bearing: {match}" for match in matches])
        
        # Distance patterns
        distance_pattern = r'\d+\.?\d*\s*(?:feet|ft|meters?|m|miles?|mi)\b'
        matches = re.findall(distance_pattern, text, re.IGNORECASE)
        patterns.extend([f"distance: {match}" for match in matches])
        
        return patterns[:20]  # Limit to first 20 patterns found
    
    def cleanup_processed_files(self, file_info: Dict) -> None:
        """Clean up temporary files created during processing"""
        try:
            # Clean up optimized images
            if 'optimized_path' in file_info and file_info['optimized_path']:
                if os.path.exists(file_info['optimized_path']):
                    os.remove(file_info['optimized_path'])
            
            # Clean up PDF conversion images
            if 'pdf_properties' in file_info:
                for image_path in file_info['pdf_properties'].get('converted_images', []):
                    if os.path.exists(image_path):
                        os.remove(image_path)
            
            logger.info("Cleaned up temporary processing files")
            
        except Exception as e:
            logger.error(f"Failed to cleanup processed files: {str(e)}")
    
    def get_file_info_summary(self, file_info: Dict) -> str:
        """Generate a human-readable summary of file processing results"""
        if file_info.get('processing_status') == 'error':
            return f"Error processing file: {file_info.get('error_message', 'Unknown error')}"
        
        summary_parts = []
        
        # Basic file info
        file_type = file_info.get('file_type', 'unknown')
        file_size_mb = file_info.get('file_size', 0) / (1024 * 1024)
        summary_parts.append(f"File type: {file_type}")
        summary_parts.append(f"Size: {file_size_mb:.1f} MB")
        
        # Type-specific info
        if file_type == 'image':
            props = file_info.get('image_properties', {})
            quality = file_info.get('quality_assessment', {})
            summary_parts.append(f"Dimensions: {props.get('width')}x{props.get('height')}")
            summary_parts.append(f"Quality score: {quality.get('quality_score', 'N/A')}")
            
        elif file_type == 'pdf':
            props = file_info.get('pdf_properties', {})
            summary_parts.append(f"Pages: {props.get('num_pages', 'N/A')}")
            
        elif file_type == 'text':
            props = file_info.get('text_properties', {})
            summary_parts.append(f"Words: {props.get('word_count', 'N/A')}")
            patterns_count = len(props.get('coordinate_patterns', []))
            summary_parts.append(f"Coordinate patterns found: {patterns_count}")
        
        ready = file_info.get('ready_for_analysis', False)
        summary_parts.append(f"Ready for analysis: {'Yes' if ready else 'No'}")
        
        return " | ".join(summary_parts) 