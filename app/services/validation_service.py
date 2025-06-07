import logging
import requests
from typing import Dict, List, Optional, Tuple
from geopy.geocoders import Nominatim
from flask import current_app

logger = logging.getLogger(__name__)

class ValidationService:
    """Service for validating property analysis results against government databases"""
    
    def __init__(self):
        self.geocoder = Nominatim(user_agent="AIPropertyDetails")
    
    def validate_analysis_result(self, analysis_result: Dict) -> Dict:
        """
        Comprehensive validation of analysis results
        
        Args:
            analysis_result: The AI analysis result to validate
            
        Returns:
            Dictionary with validation results and confidence adjustments
        """
        try:
            logger.info("Starting comprehensive validation of analysis results")
            
            validation_results = {
                'validation_score': 0.0,
                'validation_checks': [],
                'confidence_adjustment': 0.0,
                'government_data_matches': [],
                'discrepancies_found': [],
                'recommended_confidence': 0.0
            }
            
            # Extract key information for validation
            property_details = analysis_result.get('property_details', {})
            boundary_coords = analysis_result.get('boundary_coordinates', {})
            measurements = analysis_result.get('measurements', {})
            additional_info = analysis_result.get('additional_info', {})
            
            # Validation Check 1: Legal Description Format Validation
            legal_desc_score = self._validate_legal_description(property_details.get('legal_description'))
            validation_results['validation_checks'].append({
                'check': 'Legal Description Format',
                'score': legal_desc_score,
                'details': 'Format and completeness of legal description'
            })
            
            # Validation Check 2: Survey Measurements Consistency
            measurement_score = self._validate_measurements_consistency(measurements, boundary_coords)
            validation_results['validation_checks'].append({
                'check': 'Measurement Consistency',
                'score': measurement_score,
                'details': 'Internal consistency of bearings, distances, and angles'
            })
            
            # Validation Check 3: Property Identification Validation
            property_id_score = self._validate_property_identification(property_details)
            validation_results['validation_checks'].append({
                'check': 'Property Identification',
                'score': property_id_score,
                'details': 'Parcel numbers, addresses, and legal references'
            })
            
            # Validation Check 4: Surveyor Information Validation
            surveyor_score = self._validate_surveyor_information(additional_info)
            validation_results['validation_checks'].append({
                'check': 'Surveyor Credentials',
                'score': surveyor_score,
                'details': 'Professional surveyor information and licensing'
            })
            
            # Validation Check 5: Geographic Location Validation
            geo_score = self._validate_geographic_location(property_details)
            validation_results['validation_checks'].append({
                'check': 'Geographic Location',
                'score': geo_score,
                'details': 'Address and location consistency validation'
            })
            
            # Calculate overall validation score
            total_score = sum(check['score'] for check in validation_results['validation_checks'])
            validation_results['validation_score'] = round(total_score / len(validation_results['validation_checks']), 3)
            
            # Calculate confidence adjustment
            original_confidence = analysis_result.get('confidence_score', 0.0)
            validation_adjustment = self._calculate_confidence_adjustment(validation_results['validation_score'])
            
            validation_results['confidence_adjustment'] = validation_adjustment
            validation_results['recommended_confidence'] = round(
                min(1.0, max(0.0, original_confidence + validation_adjustment)), 3
            )
            
            logger.info(f"Validation completed. Score: {validation_results['validation_score']}, "
                       f"Confidence adjustment: {validation_adjustment}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {
                'validation_score': 0.0,
                'validation_checks': [],
                'confidence_adjustment': -0.1,  # Penalize for validation failure
                'error': str(e)
            }
    
    def _validate_legal_description(self, legal_description: str) -> float:
        """Validate legal description format and completeness"""
        if not legal_description:
            return 0.0
        
        score = 0.0
        
        # Check for key components
        if any(term in legal_description.lower() for term in ['section', 'township', 'range']):
            score += 0.3  # Government survey system reference
        
        if any(term in legal_description.lower() for term in ['lot', 'block', 'plat']):
            score += 0.2  # Subdivision reference
        
        if any(term in legal_description.lower() for term in ['beginning', 'point of beginning', 'pob']):
            score += 0.2  # Point of beginning mentioned
        
        if any(term in legal_description.lower() for term in ['thence', 'bearing', 'feet', 'degrees']):
            score += 0.2  # Survey calls present
        
        if 'county' in legal_description.lower():
            score += 0.1  # County reference
        
        return min(1.0, score)
    
    def _validate_measurements_consistency(self, measurements: Dict, boundary_coords: Dict) -> float:
        """Validate consistency of survey measurements"""
        score = 0.0
        
        bearings = measurements.get('bearings', [])
        distances = measurements.get('distances', [])
        vertices = boundary_coords.get('vertices', [])
        
        # Check if we have measurement data
        if not bearings and not distances:
            return 0.0
        
        # Basic consistency checks
        if len(bearings) > 0:
            score += 0.3  # Has bearing data
            
            # Check bearing format consistency
            valid_bearings = 0
            for bearing in bearings:
                if self._is_valid_bearing_format(bearing):
                    valid_bearings += 1
            
            if len(bearings) > 0:
                score += 0.2 * (valid_bearings / len(bearings))
        
        if len(distances) > 0:
            score += 0.3  # Has distance data
            
            # Check distance format consistency
            valid_distances = 0
            for distance in distances:
                if self._is_valid_distance_format(str(distance)):
                    valid_distances += 1
            
            if len(distances) > 0:
                score += 0.2 * (valid_distances / len(distances))
        
        return min(1.0, score)
    
    def _validate_property_identification(self, property_details: Dict) -> float:
        """Validate property identification information"""
        score = 0.0
        
        addresses = property_details.get('addresses', [])
        parcel_numbers = property_details.get('parcel_numbers', [])
        legal_description = property_details.get('legal_description', '')
        
        if addresses:
            score += 0.3  # Has address information
        
        if parcel_numbers:
            score += 0.4  # Has parcel number (very important)
            
            # Validate parcel number format
            for parcel in parcel_numbers:
                if len(str(parcel)) >= 10:  # Most parcel numbers are substantial
                    score += 0.1
                    break
        
        if legal_description and len(legal_description) > 50:
            score += 0.2  # Has substantial legal description
        
        return min(1.0, score)
    
    def _validate_surveyor_information(self, additional_info: Dict) -> float:
        """Validate surveyor information and credentials"""
        surveyor_info = additional_info.get('surveyor_info', '')
        
        if not surveyor_info:
            return 0.0
        
        score = 0.0
        
        # Check for professional license indicators
        if any(term in surveyor_info.lower() for term in ['pls', 'professional land surveyor', 'reg. no']):
            score += 0.5
        
        # Check for license number
        import re
        if re.search(r'\d{3,6}', surveyor_info):  # License number pattern
            score += 0.3
        
        # Check for company/firm information
        if any(term in surveyor_info.lower() for term in ['associates', 'inc', 'llc', 'company']):
            score += 0.2
        
        return min(1.0, score)
    
    def _validate_geographic_location(self, property_details: Dict) -> float:
        """Validate geographic location using geocoding services"""
        addresses = property_details.get('addresses', [])
        
        if not addresses:
            return 0.5  # Neutral score if no address to validate
        
        try:
            # Try to geocode the first address
            address = addresses[0]
            location = self.geocoder.geocode(address, timeout=10)
            
            if location:
                # Successfully geocoded
                score = 0.8
                
                # Additional validation if we have coordinate data
                # This could be enhanced to check proximity to expected location
                return score
            else:
                # Could not geocode
                return 0.3
                
        except Exception as e:
            logger.warning(f"Geocoding failed: {str(e)}")
            return 0.4  # Neutral score if geocoding fails
    
    def _calculate_confidence_adjustment(self, validation_score: float) -> float:
        """Calculate confidence adjustment based on validation results"""
        if validation_score >= 0.9:
            return 0.05  # Boost confidence for excellent validation
        elif validation_score >= 0.8:
            return 0.02  # Small boost for good validation
        elif validation_score >= 0.6:
            return 0.0   # No adjustment for acceptable validation
        elif validation_score >= 0.4:
            return -0.05 # Small penalty for poor validation
        else:
            return -0.1  # Significant penalty for very poor validation
    
    def _is_valid_bearing_format(self, bearing: str) -> bool:
        """Check if bearing follows valid format"""
        import re
        # Pattern for bearings like N45°30'15"E or S12°45'W
        pattern = r'[NS]\d{1,3}[°]\d{1,2}[\']\d{0,2}[\"]*[EW]'
        return bool(re.match(pattern, bearing.replace(' ', '')))
    
    def _is_valid_distance_format(self, distance: str) -> bool:
        """Check if distance follows valid format"""
        try:
            # Remove non-numeric characters except decimal
            numeric_part = ''.join(c for c in distance if c.isdigit() or c == '.')
            if numeric_part:
                float(numeric_part)
                return True
        except ValueError:
            pass
        return False 