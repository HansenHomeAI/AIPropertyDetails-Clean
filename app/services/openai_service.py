import base64
import json
import logging
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
            logger.info(f"Starting analysis of {document_type} document: {image_path}")
            
            # Encode image
            base64_image = self.encode_image(image_path)
            
            # Create specialized prompt based on document type
            prompt = self._create_analysis_prompt(document_type)
            
            # Call OpenAI o4-mini model
            response = self.client.chat.completions.create(
                model=current_app.config.get('OPENAI_MODEL', 'o4-mini-2025-04-16'),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=current_app.config.get('OPENAI_MAX_TOKENS', 4000),
                temperature=current_app.config.get('OPENAI_TEMPERATURE', 0.1)
            )
            
            # Parse response
            analysis_result = self._parse_analysis_response(response.choices[0].message.content)
            
            logger.info(f"Analysis completed successfully for {image_path}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Failed to analyze document {image_path}: {str(e)}")
            raise
    
    def _create_analysis_prompt(self, document_type: str) -> str:
        """Create specialized prompts for different document types"""
        
        base_prompt = """
You are an expert property analyst with advanced capabilities in reading and interpreting property documents, parcel maps, plat maps, and legal descriptions. Your primary task is to extract precise boundary coordinates and property details from the provided document.

Please analyze this document and provide a detailed JSON response with the following structure:

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
    
    def _parse_analysis_response(self, response_text: str) -> Dict:
        """Parse the JSON response from OpenAI"""
        try:
            # Try to extract JSON from the response
            # The model might return markdown code blocks, so we need to clean it
            response_text = response_text.strip()
            
            # Remove markdown code block markers if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            elif response_text.startswith('```'):
                response_text = response_text[3:]
            
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            # Parse JSON
            result = json.loads(response_text.strip())
            
            # Validate required fields
            self._validate_analysis_result(result)
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {str(e)}")
            logger.error(f"Response text: {response_text}")
            
            # Return a fallback structure
            return {
                "error": "Failed to parse analysis response",
                "raw_response": response_text,
                "boundary_coordinates": {"vertices": []},
                "property_details": {}
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