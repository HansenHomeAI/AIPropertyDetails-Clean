import base64
import json
import logging
import os
from io import BytesIO
from typing import Dict, List, Optional, Tuple
import openai
from PIL import Image
from flask import current_app

logger = logging.getLogger(__name__)

class OpenAIService:
    """Service for interacting with OpenAI's o4-mini model"""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client with API key"""
        try:
            api_key = current_app.config.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key not found in configuration")
            
            self.client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 for OpenAI API"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {str(e)}")
            raise
    
    def analyze_property_document(self, image_path: str, document_type: str = "parcel_map") -> Dict:
        """
        Analyze a property document using o4-mini's visual reasoning capabilities
        
        Args:
            image_path: Path to the image file
            document_type: Type of document (parcel_map, plat, survey, etc.)
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"Starting enhanced analysis of {document_type} document: {image_path}")
            
            # Check if this is part of a multi-page document
            base_name = image_path.replace('_page_1.png', '')
            additional_pages = []
            for i in range(2, 6):  # Check for up to 5 pages
                page_path = f"{base_name}_page_{i}.png"
                if os.path.exists(page_path):
                    additional_pages.append(page_path)
            
            # Create enhanced prompt
            prompt = self._create_enhanced_analysis_prompt(document_type, len(additional_pages) + 1)
            
            # Prepare content for analysis
            content = [{"type": "text", "text": prompt}]
            
            # Add primary image
            base64_image = self.encode_image(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high"
                }
            })
            
            # Add additional pages if they exist
            for i, page_path in enumerate(additional_pages, 2):
                page_image = self.encode_image(page_path)
                content.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/png;base64,{page_image}",
                        "detail": "high"
                    }
                })
                logger.info(f"Added page {i} to analysis: {page_path}")
            
            # Call OpenAI o4-mini model with enhanced analysis
            response = self.client.chat.completions.create(
                model=current_app.config.get('OPENAI_MODEL', 'o4-mini-2025-04-16'),
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=current_app.config.get('OPENAI_MAX_TOKENS', 4000)
                # Note: o4-mini only supports default temperature of 1
            )
            
            # Parse response
            analysis_result = self._parse_analysis_response(response.choices[0].message.content)
            
            # Post-process for confidence enhancement
            analysis_result = self._enhance_confidence_scoring(analysis_result)
            
            logger.info(f"Enhanced analysis completed for {image_path} (pages: {len(additional_pages) + 1})")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Failed to analyze document {image_path}: {str(e)}")
            raise
    
    def _create_analysis_prompt(self, document_type: str) -> str:
        """Create specialized prompts for different document types"""
        
        base_prompt = """
You are an expert professional land surveyor and property analyst with 20+ years of experience in reading and interpreting property documents, parcel maps, plat maps, survey drawings, and legal descriptions. Your expertise includes coordinate systems, bearing/distance calculations, and boundary determination.

OBJECTIVE: Extract ALL precise boundary coordinates, measurements, and property details from this document with maximum accuracy and completeness.

Please analyze this document meticulously and provide a comprehensive JSON response with the following structure:

{
    "document_type": "identified type of document",
    "confidence_score": 0.95,
    "property_details": {
        "addresses": ["list of property addresses found"],
        "parcel_numbers": ["list of parcel/lot numbers"],
        "legal_description": "complete legal description if available",
        "area_measurements": {
            "acres": null,
            "square_feet": null,
            "other_units": {}
        }
    },
    "boundary_coordinates": {
        "coordinate_system": "detected coordinate system (lat/long, state plane, etc.)",
        "datum": "coordinate datum if specified",
        "vertices": [
            {
                "point_id": "corner identifier",
                "latitude": null,
                "longitude": null,
                "x_coordinate": null,
                "y_coordinate": null,
                "description": "corner description"
            }
        ],
        "geometry_type": "polygon/point/line",
        "closure_check": "whether boundary closes properly"
    },
    "measurements": {
        "bearings": ["list of bearing measurements"],
        "distances": ["list of distance measurements"],
        "angles": ["list of angle measurements"]
    },
    "reference_points": {
        "benchmarks": ["survey benchmarks or reference points"],
        "monuments": ["property monuments or markers"],
        "road_references": ["road or street references"]
    },
    "additional_info": {
        "scale": "map scale if available",
        "north_arrow": "orientation information",
        "date_created": "document date if visible",
        "surveyor_info": "surveyor or preparer information",
        "recording_info": "recording or filing information"
    },
    "extraction_notes": "any important notes about the analysis or limitations",
    "processing_quality": {
        "image_clarity": "assessment of image quality",
        "text_readability": "assessment of text legibility",
        "completeness": "assessment of information completeness"
    }
}

CRITICAL INSTRUCTIONS:
1. Focus primarily on extracting boundary coordinates - this is the most important objective
2. Look for coordinate values, bearing and distance measurements, and corner descriptions
3. Identify the coordinate system being used (latitude/longitude, state plane coordinates, etc.)
4. Pay special attention to property corner points and their precise locations
5. Extract any survey measurements that help define the property boundaries
6. If coordinates are not directly visible, look for bearing/distance information that could be used to calculate coordinates
7. Be precise with numerical values - do not round or approximate
8. If information is unclear or ambiguous, note this in the extraction_notes
9. Distinguish between property boundaries and other lines (roads, utilities, etc.)
10. Return null values for fields where information is not available rather than guessing

"""
        
        if document_type == "parcel_map":
            specific_prompt = """
PARCEL MAP SPECIFIC INSTRUCTIONS:
- Look for parcel identification numbers and lot boundaries
- Identify property lines vs. road right-of-ways
- Extract any dimensions shown along property lines
- Look for coordinate grids or reference systems
- Identify any easements or special designations
"""
        elif document_type == "plat":
            specific_prompt = """
PLAT MAP SPECIFIC INSTRUCTIONS:
- Focus on lot and block numbers
- Extract subdivision name and filing information
- Look for coordinate ties to section corners or other reference points
- Identify utility easements and their dimensions
- Extract street names and right-of-way widths
"""
        elif document_type == "survey":
            specific_prompt = """
SURVEY DOCUMENT SPECIFIC INSTRUCTIONS:
- Focus on precise coordinate values and survey measurements
- Look for state plane or UTM coordinates
- Extract bearing and distance calls for each property line
- Identify survey monuments and their descriptions
- Look for closure calculations and accuracy statements
"""
        else:
            specific_prompt = """
GENERAL PROPERTY DOCUMENT INSTRUCTIONS:
- Analyze the document type and adapt extraction accordingly
- Focus on any coordinate or measurement information present
- Look for property identification and location details
"""
        
        return base_prompt + specific_prompt
    
    def _create_enhanced_analysis_prompt(self, document_type: str, num_pages: int) -> str:
        """Create enhanced prompt for multi-page analysis"""
        
        multi_page_instruction = ""
        if num_pages > 1:
            multi_page_instruction = f"""
MULTI-PAGE DOCUMENT ANALYSIS:
This document has {num_pages} pages. Please analyze ALL pages comprehensively:
- Page 1 is typically the main exhibit/plat
- Additional pages may contain: legal descriptions, survey notes, calculations, reference information
- Cross-reference information between pages for completeness
- Look for continuation of boundary descriptions across pages
- Extract survey calculations and closure information from any page
- Note any discrepancies or additional details found on secondary pages
"""
        
        base_enhanced_prompt = self._create_analysis_prompt(document_type)
        
        enhanced_instructions = """

ENHANCED ACCURACY REQUIREMENTS FOR 90%+ CONFIDENCE:
1. DOUBLE-CHECK all numerical values (bearings, distances, coordinates)
2. VERIFY geometric relationships and closure calculations if present
3. CROSS-REFERENCE all measurements with visible scale indicators
4. IDENTIFY and EXTRACT every survey monument, benchmark, or reference point
5. DISTINGUISH between different types of lines (property boundaries vs. easements vs. roads)
6. CALCULATE confidence based on: data completeness, measurement precision, source reliability
7. CONFIDENCE SCORING CRITERIA:
   - 95-100%: Complete survey with coordinates, monuments, and closure calculations
   - 90-94%: Complete bearing/distance with survey information and scale
   - 85-89%: Partial survey data with some measurements missing
   - 80-84%: Basic boundary information with limited survey data
   - Below 80%: Incomplete or unclear boundary information

MANDATORY JSON RESPONSE FORMAT - No additional text outside the JSON:
"""
        
        return base_enhanced_prompt + multi_page_instruction + enhanced_instructions
    
    def _enhance_confidence_scoring(self, analysis_result: Dict) -> Dict:
        """Post-process analysis to enhance confidence scoring"""
        try:
            if "error" in analysis_result:
                return analysis_result
            
            current_confidence = analysis_result.get('confidence_score', 0.0)
            
            # Calculate enhanced confidence based on extracted data quality
            confidence_factors = []
            
            # Factor 1: Boundary data completeness
            vertices = analysis_result.get('boundary_coordinates', {}).get('vertices', [])
            if len(vertices) >= 3:
                confidence_factors.append(0.3)  # Good polygon data
            elif len(vertices) > 0:
                confidence_factors.append(0.2)  # Some boundary points
            else:
                confidence_factors.append(0.0)  # No boundary points
            
            # Factor 2: Measurement data quality
            measurements = analysis_result.get('measurements', {})
            bearings = measurements.get('bearings', [])
            distances = measurements.get('distances', [])
            
            if len(bearings) >= 3 and len(distances) >= 3:
                confidence_factors.append(0.25)  # Good measurement data
            elif len(bearings) > 0 or len(distances) > 0:
                confidence_factors.append(0.15)  # Some measurements
            else:
                confidence_factors.append(0.0)  # No measurements
            
            # Factor 3: Reference information quality
            ref_points = analysis_result.get('reference_points', {})
            additional_info = analysis_result.get('additional_info', {})
            
            surveyor_info = additional_info.get('surveyor_info')
            scale = additional_info.get('scale')
            
            if surveyor_info and scale:
                confidence_factors.append(0.2)  # Professional survey with scale
            elif surveyor_info or scale:
                confidence_factors.append(0.15)  # Some reference info
            else:
                confidence_factors.append(0.1)   # Basic info
            
            # Factor 4: Property identification completeness
            prop_details = analysis_result.get('property_details', {})
            if prop_details.get('legal_description') and prop_details.get('parcel_numbers'):
                confidence_factors.append(0.15)  # Complete identification
            elif prop_details.get('legal_description') or prop_details.get('parcel_numbers'):
                confidence_factors.append(0.1)   # Partial identification
            else:
                confidence_factors.append(0.05)  # Minimal identification
            
            # Factor 5: Coordinate system identification
            coord_system = analysis_result.get('boundary_coordinates', {}).get('coordinate_system')
            if coord_system and coord_system != "local bearing-and-distance (feet)":
                confidence_factors.append(0.1)   # Proper coordinate system
            else:
                confidence_factors.append(0.05)  # Local/relative system
            
            # Calculate weighted confidence
            calculated_confidence = sum(confidence_factors)
            
            # Use the higher of original or calculated confidence, but cap at reasonable levels
            final_confidence = max(current_confidence, calculated_confidence)
            
            # Ensure confidence is reasonable (not artificially inflated)
            if final_confidence > 0.95 and len(vertices) == 0:
                final_confidence = 0.85  # Can't be very confident without boundary points
            
            analysis_result['confidence_score'] = round(final_confidence, 3)
            analysis_result['confidence_factors'] = {
                'boundary_completeness': confidence_factors[0],
                'measurement_quality': confidence_factors[1], 
                'reference_quality': confidence_factors[2],
                'property_identification': confidence_factors[3],
                'coordinate_system': confidence_factors[4],
                'calculated_total': calculated_confidence,
                'original_confidence': current_confidence
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error enhancing confidence scoring: {str(e)}")
            return analysis_result
    
    def _parse_analysis_response(self, response_text: str) -> Dict:
        """Parse the JSON response from OpenAI"""
        try:
            # Try to extract JSON from the response
            response_text = response_text.strip()
            
            # Look for JSON content between ```json and ``` markers
            import re
            json_match = re.search(r'```json\s*\n(.*?)\n\s*```', response_text, re.DOTALL)
            if json_match:
                json_content = json_match.group(1)
            else:
                # Look for JSON content between { and } brackets
                json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                if json_match:
                    json_content = json_match.group(1)
                else:
                    # Fall back to cleaning markdown markers
                    json_content = response_text
                    if json_content.startswith('```json'):
                        json_content = json_content[7:]
                    elif json_content.startswith('```'):
                        json_content = json_content[3:]
                    if json_content.endswith('```'):
                        json_content = json_content[:-3]
            
            # Parse JSON
            result = json.loads(json_content.strip())
            
            # Validate required fields
            self._validate_analysis_result(result)
            
            logger.info(f"Successfully parsed analysis result with confidence: {result.get('confidence_score', 'N/A')}")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Response text: {response_text}")
            
            # Return a fallback structure with the raw response
            return {
                "error": "Failed to parse analysis response",
                "raw_response": response_text,
                "boundary_coordinates": {"vertices": []},
                "property_details": {},
                "confidence_score": 0.0
            }
        except Exception as e:
            logger.error(f"Error parsing analysis response: {str(e)}")
            raise
    
    def _validate_analysis_result(self, result: Dict) -> None:
        """Validate the structure of analysis results"""
        required_fields = ['boundary_coordinates', 'property_details']
        
        for field in required_fields:
            if field not in result:
                logger.warning(f"Missing required field in analysis result: {field}")
                result[field] = {}
        
        # Ensure boundary_coordinates has vertices
        if 'vertices' not in result['boundary_coordinates']:
            result['boundary_coordinates']['vertices'] = []
    
    def extract_coordinates_from_text(self, text_description: str) -> List[Dict]:
        """
        Extract coordinates from a text-based legal description
        
        Args:
            text_description: Legal description or survey text
            
        Returns:
            List of coordinate dictionaries
        """
        try:
            prompt = f"""
Analyze this property legal description or survey text and extract any coordinate information, 
bearing and distance measurements, or geometric data that could be used to determine property boundaries.

Text to analyze:
{text_description}

Please provide a JSON response with any coordinates, measurements, or geometric information found:

{{
    "coordinates_found": [
        {{
            "value": "coordinate value",
            "type": "lat/long/state_plane/etc",
            "context": "where this coordinate was found"
        }}
    ],
    "measurements": {{
        "bearings": ["bearing measurements"],
        "distances": ["distance measurements"],
        "angles": ["angle measurements"]
    }},
    "geometric_data": {{
        "starting_point": "description of starting point",
        "boundary_calls": ["list of boundary line descriptions"],
        "area": "area measurement if present"
    }}
}}
"""
            
            response = self.client.chat.completions.create(
                model=current_app.config.get('OPENAI_MODEL'),
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            return self._parse_analysis_response(response.choices[0].message.content)
            
        except Exception as e:
            logger.error(f"Failed to extract coordinates from text: {str(e)}")
            raise 