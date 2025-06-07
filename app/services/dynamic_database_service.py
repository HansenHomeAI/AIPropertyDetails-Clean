"""
Dynamic Database Service - Discover and interact with local government property databases
Uses o4-mini's web search capabilities to find relevant databases for any location
"""

import logging
import requests
import json
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
import time

logger = logging.getLogger(__name__)

class DynamicDatabaseService:
    """Service for dynamically discovering and querying government property databases"""
    
    def __init__(self, openai_service):
        self.openai_service = openai_service
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Cache discovered databases to avoid repeated searches
        self.database_cache = {}
    
    def discover_and_search_databases(self, property_details: Dict) -> Dict:
        """
        Main method: Discover local government databases and search for property data
        Uses o4-mini to intelligently find and navigate government websites
        """
        
        logger.info("Starting dynamic database discovery and search")
        
        result = {
            'coordinates_found': False,
            'source': None,
            'vertices': [],
            'confidence': 0.0,
            'search_results': {},
            'discovered_databases': []
        }
        
        # Step 1: Extract location information
        location_info = self._extract_location_details(property_details)
        if not location_info:
            logger.warning("Could not extract location information")
            return result
        
        logger.info(f"Location extracted: {location_info}")
        
        # Step 2: Use o4-mini to discover relevant government databases
        databases = self._discover_government_databases(location_info)
        result['discovered_databases'] = databases
        
        if not databases:
            logger.warning("No government databases discovered")
            return result
        
        # Step 3: Search each discovered database
        for db in databases:
            logger.info(f"Searching database: {db['name']} at {db['url']}")
            
            search_result = self._search_database(db, property_details, location_info)
            
            if search_result['coordinates_found']:
                result.update(search_result)
                result['source'] = db['name']
                result['confidence'] = search_result['confidence']
                logger.info(f"SUCCESS: Found coordinates in {db['name']}")
                return result
        
        logger.warning("No coordinates found in any discovered database")
        return result
    
    def _extract_location_details(self, property_details: Dict) -> Optional[Dict]:
        """Extract detailed location information from property details"""
        
        location = {
            'addresses': property_details.get('addresses', []),
            'city': None,
            'county': None,
            'state': None,
            'country': None,
            'postal_code': None
        }
        
        # Parse addresses to extract location components
        for address in location['addresses']:
            parsed = self._parse_address_with_ai(address)
            if parsed:
                location.update(parsed)
                break
        
        # Extract from legal description if addresses aren't sufficient
        legal_desc = property_details.get('legal_description', '')
        if legal_desc and not location['county']:
            location_from_legal = self._extract_location_from_legal_desc(legal_desc)
            if location_from_legal:
                location.update(location_from_legal)
        
        return location if any(v for v in location.values() if v) else None
    
    def _parse_address_with_ai(self, address: str) -> Optional[Dict]:
        """Use o4-mini to parse address into standardized components"""
        
        prompt = f"""Parse this property address into its components. Return ONLY a JSON object:

Address: "{address}"

Return format:
{{"city": "city_name", "county": "county_name", "state": "state_name", "country": "country_name", "postal_code": "zip_code"}}

Use null for missing values. Be precise with official names."""
        
        try:
            response = self.openai_service.call_text_api([{
                "role": "user",
                "content": prompt
            }])
            
            result = json.loads(response.strip())
            return {k: v for k, v in result.items() if v}
            
        except Exception as e:
            logger.warning(f"AI address parsing failed: {str(e)}")
            return None
    
    def _extract_location_from_legal_desc(self, legal_desc: str) -> Optional[Dict]:
        """Extract location info from legal description using AI"""
        
        prompt = f"""Extract location information from this legal property description:

"{legal_desc}"

Return ONLY a JSON object:
{{"city": "...", "county": "...", "state": "...", "country": "..."}}

Use null for missing values."""
        
        try:
            response = self.openai_service.call_text_api([{
                "role": "user", 
                "content": prompt
            }])
            
            result = json.loads(response.strip())
            return {k: v for k, v in result.items() if v}
            
        except Exception as e:
            logger.warning(f"Legal description parsing failed: {str(e)}")
            return None
    
    def _discover_government_databases(self, location_info: Dict) -> List[Dict]:
        """Use o4-mini to discover relevant government property databases"""
        
        # Check cache first
        cache_key = f"{location_info.get('county', '')}-{location_info.get('state', '')}-{location_info.get('country', '')}"
        if cache_key in self.database_cache:
            logger.info("Using cached database discovery results")
            return self.database_cache[cache_key]
        
        location_str = self._format_location_for_search(location_info)
        
        prompt = f"""Find official government property/parcel databases for: {location_str}

Search for:
1. County assessor websites
2. GIS/mapping portals  
3. Property search databases
4. Parcel viewer applications

Return ONLY a JSON array:
[
    {{
        "name": "Official Name",
        "url": "https://exact-url.com/property-search",
        "type": "assessor",
        "jurisdiction": "county",
        "search_method": "address"
    }}
]

Find REAL working government websites. Verify URLs exist. Types: assessor, gis, parcel_viewer, property_search"""
        
        try:
            response = self.openai_service.call_text_api([{
                "role": "user",
                "content": prompt
            }])
            
            # Debug: Log the raw response
            logger.info(f"Raw AI response for database discovery: {response[:500]}...")
            
            # Try to extract JSON from the response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_content = response[json_start:json_end]
                databases = json.loads(json_content)
            else:
                logger.warning("No JSON array found in response")
                return []
            
            # Validate and filter results
            valid_databases = []
            for db in databases:
                if self._validate_database_info(db):
                    valid_databases.append(db)
            
            # Cache the results
            self.database_cache[cache_key] = valid_databases
            
            logger.info(f"Discovered {len(valid_databases)} valid government databases")
            return valid_databases
            
        except Exception as e:
            logger.error(f"Database discovery failed: {str(e)}")
            return []
    
    def _format_location_for_search(self, location_info: Dict) -> str:
        """Format location info for search queries"""
        
        parts = []
        for key in ['city', 'county', 'state', 'country']:
            if location_info.get(key):
                parts.append(location_info[key])
        
        return ', '.join(parts)
    
    def _validate_database_info(self, db: Dict) -> bool:
        """Validate that database info is complete and URL is reachable"""
        
        required_fields = ['name', 'url', 'type']
        if not all(field in db for field in required_fields):
            return False
        
        # Quick URL validation
        try:
            response = self.session.head(db['url'], timeout=10, allow_redirects=True)
            return response.status_code < 400
        except:
            logger.warning(f"URL validation failed for {db.get('url')}")
            return False
    
    def _search_database(self, database: Dict, property_details: Dict, 
                        location_info: Dict) -> Dict:
        """Search a specific government database for property information"""
        
        result = {
            'coordinates_found': False,
            'vertices': [],
            'confidence': 0.0,
            'search_results': {}
        }
        
        try:
            # Use o4-mini to understand the database interface and search
            search_result = self._ai_guided_database_search(
                database, property_details, location_info
            )
            
            if search_result['success']:
                # Extract coordinates from the search results
                coordinates = self._extract_coordinates_from_results(
                    search_result['data'], database
                )
                
                if coordinates:
                    result['coordinates_found'] = True
                    result['vertices'] = coordinates
                    result['confidence'] = 0.90  # High confidence for government data
                    result['search_results'] = search_result['data']
            
        except Exception as e:
            logger.warning(f"Database search failed for {database['name']}: {str(e)}")
        
        return result
    
    def _ai_guided_database_search(self, database: Dict, property_details: Dict, 
                                  location_info: Dict) -> Dict:
        """Use o4-mini to navigate and search the government database"""
        
        # Get the database homepage
        try:
            response = self.session.get(database['url'], timeout=15)
            page_content = response.text[:10000]  # Limit content for AI processing
        except Exception as e:
            return {'success': False, 'error': str(e)}
        
        # Extract search terms
        search_terms = self._extract_search_terms(property_details)
        
        prompt = f"""Analyze this government property database webpage and provide search strategy:

DATABASE: {database['name']}
URL: {database['url']}

WEBPAGE CONTENT (first 10,000 chars):
{page_content}

PROPERTY TO SEARCH FOR:
{search_terms}

Analyze the webpage and return ONLY a JSON object:
{{
    "search_form_found": true/false,
    "search_fields": ["address", "parcel_number", "etc"],
    "search_strategy": "description of how to search",
    "search_url": "URL for search endpoint",
    "search_parameters": {{"field_name": "search_value"}}
}}

Focus on finding working search forms, input fields, and URLs."""
        
        try:
            ai_response = self.openai_service.call_text_api([{
                "role": "user",
                "content": prompt
            }])
            
            search_strategy = json.loads(ai_response.strip())
            
            if search_strategy.get('search_form_found'):
                # Execute the search based on AI guidance
                return self._execute_database_search(database, search_strategy)
            else:
                return {'success': False, 'error': 'No search form found'}
            
        except Exception as e:
            return {'success': False, 'error': f'AI search guidance failed: {str(e)}'}
    
    def _extract_search_terms(self, property_details: Dict) -> str:
        """Extract key search terms from property details"""
        
        terms = []
        
        if property_details.get('addresses'):
            terms.append(f"Address: {property_details['addresses'][0]}")
        
        if property_details.get('parcel_numbers'):
            terms.append(f"Parcel Number: {property_details['parcel_numbers'][0]}")
        
        if property_details.get('legal_description'):
            # Truncate long legal descriptions
            legal_desc = property_details['legal_description'][:500]
            terms.append(f"Legal Description: {legal_desc}")
        
        return '\n'.join(terms)
    
    def _execute_database_search(self, database: Dict, search_strategy: Dict) -> Dict:
        """Execute the actual database search based on AI strategy"""
        
        try:
            search_url = search_strategy.get('search_url', database['url'])
            search_params = search_strategy.get('search_parameters', {})
            
            # Perform the search
            if search_strategy.get('method', 'GET').upper() == 'POST':
                response = self.session.post(search_url, data=search_params, timeout=15)
            else:
                response = self.session.get(search_url, params=search_params, timeout=15)
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'data': {
                        'html_content': response.text[:20000],  # Limit content
                        'search_url': response.url,
                        'search_params': search_params
                    }
                }
            else:
                return {'success': False, 'error': f'Search failed with status {response.status_code}'}
                
        except Exception as e:
            return {'success': False, 'error': f'Search execution failed: {str(e)}'}
    
    def _extract_coordinates_from_results(self, search_data: Dict, database: Dict) -> List[Dict]:
        """Extract property coordinates from search results using AI"""
        
        html_content = search_data.get('html_content', '')
        if not html_content:
            return []
        
        prompt = f"""Extract property boundary coordinates from this government database search result:

DATABASE: {database['name']}

HTML CONTENT:
{html_content[:15000]}

Look for:
1. Latitude/longitude coordinates
2. Property boundary vertices
3. Parcel geometry data
4. GIS coordinate information
5. Survey coordinate points

Return ONLY a JSON array of coordinate objects:
[
    {{
        "latitude": 47.123456,
        "longitude": -122.654321,
        "point_id": "corner_1",
        "description": "Northwest corner"
    }}
]

Return empty array [] if no coordinates found. Coordinates must be valid lat/lng values."""
        
        try:
            response = self.openai_service.call_text_api([{
                "role": "user",
                "content": prompt
            }])
            
            coordinates = json.loads(response.strip())
            
            # Validate coordinates
            valid_coords = []
            for coord in coordinates:
                if (isinstance(coord.get('latitude'), (int, float)) and 
                    isinstance(coord.get('longitude'), (int, float)) and
                    -90 <= coord['latitude'] <= 90 and
                    -180 <= coord['longitude'] <= 180):
                    valid_coords.append(coord)
            
            logger.info(f"Extracted {len(valid_coords)} valid coordinates from {database['name']}")
            return valid_coords
            
        except Exception as e:
            logger.warning(f"Coordinate extraction failed: {str(e)}")
            return [] 