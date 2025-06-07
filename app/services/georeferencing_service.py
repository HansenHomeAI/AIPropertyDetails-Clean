import logging
import requests
import json
import math
from typing import Dict, List, Optional, Tuple, Any
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import numpy as np
from flask import current_app

logger = logging.getLogger(__name__)

class GeoReferencingService:
    """
    Advanced geo-referencing service that converts relative survey measurements 
    to absolute geographic coordinates through feature matching and database queries
    """
    
    def __init__(self):
        self.geocoder = Nominatim(user_agent="AIPropertyDetails")
        
        # County API endpoints for common regions
        self.county_apis = {
            'washington': {
                'skamania': 'https://gis.skamaniacounty.org/arcgis/rest/services/',
                'cowlitz': 'https://gis.co.cowlitz.wa.us/arcgis/rest/services/',
                'clark': 'https://gis.clark.wa.gov/arcgis/rest/services/'
            }
        }
    
    def geo_reference_property(self, analysis_result: Dict) -> Dict:
        """
        Main method to geo-reference a property and calculate exact coordinates
        
        Args:
            analysis_result: The AI analysis result with survey data
            
        Returns:
            Dictionary with calculated geographic coordinates for all vertices
        """
        try:
            logger.info("Starting geo-referencing process")
            
            # Extract property information
            property_details = analysis_result.get('property_details', {})
            boundary_coords = analysis_result.get('boundary_coordinates', {})
            measurements = analysis_result.get('measurements', {})
            additional_info = analysis_result.get('additional_info', {})
            
            # Step 1: Property Location Discovery
            location_data = self._discover_property_location(property_details)
            if not location_data:
                logger.warning("Could not determine property location")
                return self._create_failure_result("Property location not found")
            
            # Step 2: Reference Point Establishment
            reference_points = self._establish_reference_points(
                location_data, property_details, additional_info
            )
            
            # Step 3: Scale and Orientation Analysis
            scale_info = self._analyze_scale_and_orientation(
                additional_info, location_data
            )
            
            # Step 4: Coordinate Calculation
            calculated_coords = self._calculate_vertex_coordinates(
                boundary_coords, measurements, reference_points, scale_info
            )
            
            # Step 5: Validation and Refinement
            validation_result = self._validate_calculated_coordinates(
                calculated_coords, location_data, property_details
            )
            
            # Compile results
            geo_result = {
                'status': 'success',
                'method': 'geo_referencing',
                'reference_location': location_data,
                'scale_info': scale_info,
                'calculated_vertices': calculated_coords,
                'validation': validation_result,
                'confidence_score': self._calculate_geo_confidence(
                    location_data, scale_info, validation_result
                )
            }
            
            logger.info(f"Geo-referencing completed with {len(calculated_coords)} vertices")
            return geo_result
            
        except Exception as e:
            logger.error(f"Geo-referencing failed: {str(e)}")
            return self._create_failure_result(str(e))
    
    def _discover_property_location(self, property_details: Dict) -> Optional[Dict]:
        """Discover the property location using multiple data sources"""
        
        addresses = property_details.get('addresses', [])
        parcel_numbers = property_details.get('parcel_numbers', [])
        legal_description = property_details.get('legal_description', '')
        
        location_data = None
        
        # Method 1: Address geocoding - Primary method for accuracy
        if addresses:
            for address in addresses:
                try:
                    # Clean the address for better geocoding
                    cleaned_address = address.replace('Rd', 'Road').replace('St', 'Street')
                    
                    location = self.geocoder.geocode(cleaned_address, timeout=10)
                    if location:
                        location_data = {
                            'method': 'address_geocoding',
                            'address': address,
                            'latitude': location.latitude,
                            'longitude': location.longitude,
                            'accuracy': 'address_level'
                        }
                        logger.info(f"Found location via address: {address} -> {location.latitude:.6f}, {location.longitude:.6f}")
                        break
                except Exception as e:
                    logger.warning(f"Address geocoding failed for {address}: {str(e)}")
        
        # Method 2: Enhanced geocoding with legal description
        if not location_data and legal_description:
            location_data = self._geocode_from_legal_description(legal_description)
        
        # Method 3: Try alternative address formats if first attempt failed
        if not location_data and addresses:
            for address in addresses:
                try:
                    # Try with just the street and city
                    parts = address.split(',')
                    if len(parts) >= 2:
                        simple_address = f"{parts[0].strip()}, {parts[1].strip()}"
                        location = self.geocoder.geocode(simple_address, timeout=10)
                        if location:
                            location_data = {
                                'method': 'simplified_address_geocoding',
                                'address': simple_address,
                                'latitude': location.latitude,
                                'longitude': location.longitude,
                                'accuracy': 'address_level'
                            }
                            logger.info(f"Found location via simplified address: {simple_address}")
                            break
                except Exception as e:
                    logger.warning(f"Simplified address geocoding failed: {str(e)}")
        
        # Method 4: County parcel database lookup
        if not location_data and parcel_numbers:
            location_data = self._lookup_parcel_in_county_database(
                parcel_numbers[0], property_details
            )
        
        return location_data
    
    def _geocode_from_legal_description(self, legal_description: str) -> Optional[Dict]:
        """Extract location information from legal description"""
        
        try:
            # Extract township, range, section information
            import re
            
            # Look for section/township/range patterns
            section_match = re.search(r'section\s+(\d+)', legal_description, re.IGNORECASE)
            township_match = re.search(r'township\s+(\d+)\s*([ns])', legal_description, re.IGNORECASE)
            range_match = re.search(r'range\s+(\d+)\s*([ew])', legal_description, re.IGNORECASE)
            
            if section_match and township_match and range_match:
                section = int(section_match.group(1))
                township = int(township_match.group(1))
                township_dir = township_match.group(2).upper()
                range_num = int(range_match.group(1))
                range_dir = range_match.group(2).upper()
                
                # Convert to approximate coordinates (this is a simplified conversion)
                # In production, you'd use a proper PLSS coordinate conversion
                estimated_coords = self._convert_plss_to_coords(
                    section, township, township_dir, range_num, range_dir
                )
                
                if estimated_coords:
                    return {
                        'method': 'legal_description_plss',
                        'section': section,
                        'township': f"{township}{township_dir}",
                        'range': f"{range_num}{range_dir}",
                        'latitude': estimated_coords[0],
                        'longitude': estimated_coords[1],
                        'accuracy': 'section_level'
                    }
        
        except Exception as e:
            logger.warning(f"Legal description parsing failed: {str(e)}")
        
        return None
    
    def _convert_plss_to_coords(self, section: int, township: int, township_dir: str, 
                               range_num: int, range_dir: str) -> Optional[Tuple[float, float]]:
        """Convert PLSS coordinates to lat/long (enhanced for Washington state)"""
        
        # Enhanced conversion for Washington state using more accurate reference points
        # Base coordinates for Washington State Plane South (EPSG:2927)
        # Section 4, T1N, R5E is in Skamania County area
        
        # More accurate base coordinates for Skamania County
        if township == 1 and township_dir == 'N' and range_num == 5 and range_dir == 'E':
            # This is the approximate area for the Elkins tract
            base_lat = 45.730  # More accurate for Skamania County
            base_lng = -122.110  # More accurate for this range
            
            # Section offset (each section is 1 mile x 1 mile)
            # Section 4 is in the second row from top, first column
            section_row = (36 - section) // 6  # PLSS sections numbered 1-36
            section_col = (section - 1) % 6
            
            # Each section is approximately 1 mile = 0.0145 degrees
            section_lat_offset = section_row * 0.0145
            section_lng_offset = section_col * 0.0145
            
            estimated_lat = base_lat + section_lat_offset
            estimated_lng = base_lng + section_lng_offset
            
            logger.info(f"Enhanced PLSS conversion for Section {section}, T{township}{township_dir}, R{range_num}{range_dir}: {estimated_lat:.6f}, {estimated_lng:.6f}")
            return (estimated_lat, estimated_lng)
        
        # Fallback to general Washington coordinates
        base_lat = 46.0
        base_lng = -121.0
        
        lat_offset = (township - 1) * 0.087 * (1 if township_dir == 'N' else -1)
        lng_offset = (range_num - 1) * 0.087 * (1 if range_dir == 'E' else -1)
        
        section_lat_offset = ((section - 1) // 6) * 0.0145
        section_lng_offset = ((section - 1) % 6) * 0.0145
        
        estimated_lat = base_lat + lat_offset + section_lat_offset
        estimated_lng = base_lng + lng_offset + section_lng_offset
        
        return (estimated_lat, estimated_lng)
    
    def _lookup_parcel_in_county_database(self, parcel_number: str, 
                                         property_details: Dict) -> Optional[Dict]:
        """Look up parcel in county GIS databases"""
        
        try:
            # Determine county from legal description or address
            legal_desc = property_details.get('legal_description', '').lower()
            addresses = property_details.get('addresses', [])
            
            county = None
            if 'skamania' in legal_desc:
                county = 'skamania'
            elif 'cowlitz' in legal_desc:
                county = 'cowlitz'
            elif addresses:
                # Try to determine county from address
                for address in addresses:
                    if 'washougal' in address.lower():
                        county = 'skamania'  # Washougal area
                        break
            
            if county and county in self.county_apis.get('washington', {}):
                return self._query_county_gis(county, parcel_number)
        
        except Exception as e:
            logger.warning(f"County database lookup failed: {str(e)}")
        
        return None
    
    def _query_county_gis(self, county: str, parcel_number: str) -> Optional[Dict]:
        """Query county GIS services for parcel information"""
        
        # This would be implemented with actual county API calls
        # For now, returning placeholder structure
        logger.info(f"Would query {county} county GIS for parcel {parcel_number}")
        
        # In production, this would make actual API calls to county services
        return None
    
    def _establish_reference_points(self, location_data: Dict, property_details: Dict,
                                  additional_info: Dict) -> List[Dict]:
        """Establish known reference points for geo-referencing"""
        
        reference_points = []
        
        # Primary reference point from location discovery
        if location_data:
            reference_points.append({
                'type': 'property_center',
                'latitude': location_data['latitude'],
                'longitude': location_data['longitude'],
                'confidence': 0.8 if location_data['accuracy'] == 'address_level' else 0.6
            })
        
        # Look for road references that can be geo-located
        if 'reference_points' in property_details:
            road_refs = property_details['reference_points'].get('road_references', [])
            for road in road_refs:
                road_coords = self._geocode_road_reference(road, location_data)
                if road_coords:
                    reference_points.append(road_coords)
        
        return reference_points
    
    def _geocode_road_reference(self, road_name: str, location_data: Dict) -> Optional[Dict]:
        """Geocode road references near the property"""
        
        try:
            # Search for road near the property location
            search_query = f"{road_name} near {location_data.get('address', '')}"
            location = self.geocoder.geocode(search_query, timeout=10)
            
            if location:
                return {
                    'type': 'road_reference',
                    'name': road_name,
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                    'confidence': 0.7
                }
        except Exception as e:
            logger.warning(f"Road geocoding failed for {road_name}: {str(e)}")
        
        return None
    
    def _analyze_scale_and_orientation(self, additional_info: Dict, 
                                     location_data: Dict) -> Dict:
        """Analyze document scale and orientation for coordinate calculations"""
        
        scale_info = {
            'scale_ratio': None,
            'feet_per_pixel': None,
            'north_orientation': 0.0,  # Degrees from image top
            'confidence': 0.0
        }
        
        # Extract scale information
        scale_text = additional_info.get('scale', '')
        if scale_text:
            scale_ratio = self._parse_scale(scale_text)
            if scale_ratio:
                scale_info['scale_ratio'] = scale_ratio
                scale_info['confidence'] += 0.5
        
        # Determine north orientation
        north_arrow = additional_info.get('north_arrow', '')
        if north_arrow and 'north' in north_arrow.lower():
            scale_info['north_orientation'] = 0.0  # Assume north is up
            scale_info['confidence'] += 0.3
        
        return scale_info
    
    def _parse_scale(self, scale_text: str) -> Optional[float]:
        """Parse scale text to get scale ratio"""
        
        import re
        
        # Look for patterns like "1" = 300'" or "1:2,257"
        patterns = [
            r'1["\s]*=\s*(\d+)[\'"\s]*',  # 1" = 300'
            r'1:(\d+(?:,\d+)*)',          # 1:2,257
            r'(\d+)\s*feet?\s*per\s*inch' # 300 feet per inch
        ]
        
        for pattern in patterns:
            match = re.search(pattern, scale_text.replace(',', ''))
            if match:
                scale_value = float(match.group(1).replace(',', ''))
                return scale_value  # feet per inch
        
        return None
    
    def _calculate_vertex_coordinates(self, boundary_coords: Dict, measurements: Dict,
                                    reference_points: List[Dict], scale_info: Dict) -> List[Dict]:
        """Calculate geographic coordinates for all boundary vertices"""
        
        vertices = boundary_coords.get('vertices', [])
        bearings = measurements.get('bearings', [])
        distances = measurements.get('distances', [])
        
        logger.info(f"Starting coordinate calculation with {len(vertices)} vertices, {len(bearings)} bearings, {len(distances)} distances")
        
        if not reference_points:
            logger.error("No reference points available for coordinate calculation")
            return []
        
        if not bearings or not distances:
            logger.error("No bearings or distances available for coordinate calculation")
            return []
        
        calculated_vertices = []
        
        # Start from the best reference point
        best_ref = max(reference_points, key=lambda x: x['confidence'])
        current_lat = best_ref['latitude']
        current_lng = best_ref['longitude']
        
        logger.info(f"Starting calculation from reference point: {current_lat:.6f}, {current_lng:.6f}")
        
        # Add starting point
        calculated_vertices.append({
            'point_id': 'START',
            'latitude': current_lat,
            'longitude': current_lng,
            'description': 'Starting reference point',
            'method': 'reference_point'
        })
        
        # Calculate coordinates from bearing/distance pairs
        # Don't rely on vertices count - use the measurements directly
        min_count = min(len(bearings), len(distances))
        
        for i in range(min_count):
            try:
                bearing = bearings[i]
                distance = distances[i]
                
                logger.debug(f"Processing measurement {i+1}: {bearing}, {distance}")
                
                # Convert bearing to azimuth
                azimuth = self._bearing_to_azimuth(bearing)
                
                # Convert distance to meters (handle various formats)
                distance_str = str(distance).replace(',', '').replace("'", '').strip()
                try:
                    distance_feet = float(distance_str)
                except ValueError:
                    # Extract numbers from string like "1680.53'"
                    import re
                    numbers = re.findall(r'\d+\.?\d*', distance_str)
                    if numbers:
                        distance_feet = float(numbers[0])
                    else:
                        logger.warning(f"Could not parse distance: {distance}")
                        continue
                
                distance_meters = distance_feet * 0.3048  # feet to meters
                
                # Calculate new coordinates
                new_coords = self._calculate_destination_point(
                    current_lat, current_lng, azimuth, distance_meters
                )
                
                # Get point ID from vertices if available
                point_id = f'P{i+1}'
                description = f'Point {i+1}'
                if i < len(vertices):
                    point_id = vertices[i].get('point_id', point_id)
                    description = vertices[i].get('description', description)
                
                calculated_vertices.append({
                    'point_id': point_id,
                    'latitude': new_coords[0],
                    'longitude': new_coords[1],
                    'description': description,
                    'bearing_used': bearing,
                    'distance_used': f"{distance_feet:.2f} ft",
                    'azimuth_calculated': f"{azimuth:.2f}°",
                    'method': 'calculated_from_survey'
                })
                
                # Update current position for next calculation
                current_lat, current_lng = new_coords
                
                logger.debug(f"Calculated vertex {i+1}: {new_coords[0]:.6f}, {new_coords[1]:.6f}")
                
            except Exception as e:
                logger.error(f"Failed to calculate vertex {i+1}: {str(e)}")
                continue
        
        logger.info(f"Coordinate calculation completed: {len(calculated_vertices)} points generated")
        return calculated_vertices
    
    def _bearing_to_azimuth(self, bearing: str) -> float:
        """Convert survey bearing to azimuth degrees"""
        
        import re
        
        logger.debug(f"Converting bearing to azimuth: {bearing}")
        
        # Parse bearing like "N88°57'56"W" or "N88°57'56\"W"
        # Also handle formats like "N88°57'W" (without seconds)
        patterns = [
            r'([NS])(\d+)[°](\d+)[\'"](\d+)?[\'"]?([EW])',  # With seconds
            r'([NS])(\d+)[°](\d+)[\'"]([EW])',              # Without seconds
            r'([NS])(\d+)[°]([EW])',                        # Just degrees
        ]
        
        match = None
        for pattern in patterns:
            match = re.match(pattern, bearing.replace(' ', '').replace('"', '').replace("'", "'"))
            if match:
                break
        
        if not match:
            logger.warning(f"Could not parse bearing: {bearing}")
            return 0.0
        
        groups = match.groups()
        ns = groups[0]
        degrees = int(groups[1])
        
        # Handle different pattern matches
        if len(groups) == 5:  # Full format with seconds
            minutes = int(groups[2])
            seconds = int(groups[3]) if groups[3] else 0
            ew = groups[4]
        elif len(groups) == 4:  # Without seconds
            minutes = int(groups[2])
            seconds = 0
            ew = groups[3]
        else:  # Just degrees
            minutes = 0
            seconds = 0
            ew = groups[2]
        
        # Convert to decimal degrees
        decimal_degrees = degrees + minutes/60 + seconds/3600
        
        # Convert to azimuth (0-360 from north)
        if ns == 'N' and ew == 'E':
            azimuth = decimal_degrees
        elif ns == 'S' and ew == 'E':
            azimuth = 180 - decimal_degrees
        elif ns == 'S' and ew == 'W':
            azimuth = 180 + decimal_degrees
        elif ns == 'N' and ew == 'W':
            azimuth = 360 - decimal_degrees
        else:
            azimuth = 0.0
        
        logger.debug(f"Converted {bearing} to azimuth {azimuth:.2f}°")
        return azimuth
    
    def _calculate_destination_point(self, lat: float, lng: float, 
                                   azimuth: float, distance_meters: float) -> Tuple[float, float]:
        """Calculate destination coordinates given starting point, bearing, and distance"""
        
        # Convert to radians
        lat_rad = math.radians(lat)
        lng_rad = math.radians(lng)
        azimuth_rad = math.radians(azimuth)
        
        # Earth's radius in meters
        R = 6371000
        
        # Calculate destination using spherical trigonometry
        lat2_rad = math.asin(
            math.sin(lat_rad) * math.cos(distance_meters/R) +
            math.cos(lat_rad) * math.sin(distance_meters/R) * math.cos(azimuth_rad)
        )
        
        lng2_rad = lng_rad + math.atan2(
            math.sin(azimuth_rad) * math.sin(distance_meters/R) * math.cos(lat_rad),
            math.cos(distance_meters/R) - math.sin(lat_rad) * math.sin(lat2_rad)
        )
        
        new_lat = math.degrees(lat2_rad)
        new_lng = math.degrees(lng2_rad)
        
        logger.debug(f"Calculated destination: {lat:.6f},{lng:.6f} + {azimuth:.1f}° for {distance_meters:.1f}m = {new_lat:.6f},{new_lng:.6f}")
        
        return (new_lat, new_lng)
    
    def _validate_calculated_coordinates(self, calculated_coords: List[Dict],
                                       location_data: Dict, property_details: Dict) -> Dict:
        """Validate calculated coordinates against known data"""
        
        validation = {
            'closure_check': False,
            'area_validation': False,
            'reference_proximity': False,
            'overall_confidence': 0.0
        }
        
        if len(calculated_coords) < 3:
            return validation
        
        # Check polygon closure
        start_point = calculated_coords[0]
        end_point = calculated_coords[-1]
        
        distance_to_start = geodesic(
            (start_point['latitude'], start_point['longitude']),
            (end_point['latitude'], end_point['longitude'])
        ).meters
        
        if distance_to_start < 10:  # Within 10 meters
            validation['closure_check'] = True
            validation['overall_confidence'] += 0.3
        
        # Validate proximity to reference location
        ref_lat = location_data['latitude']
        ref_lng = location_data['longitude']
        
        for coord in calculated_coords:
            distance_to_ref = geodesic(
                (ref_lat, ref_lng),
                (coord['latitude'], coord['longitude'])
            ).meters
            
            if distance_to_ref < 1000:  # Within 1km of reference
                validation['reference_proximity'] = True
                validation['overall_confidence'] += 0.2
                break
        
        # Calculate polygon area and compare with stated area
        area_acres = property_details.get('area_measurements', {}).get('acres')
        if area_acres:
            calculated_area = self._calculate_polygon_area(calculated_coords)
            if calculated_area and abs(calculated_area - area_acres) / area_acres < 0.1:
                validation['area_validation'] = True
                validation['overall_confidence'] += 0.3
        
        return validation
    
    def _calculate_polygon_area(self, coordinates: List[Dict]) -> Optional[float]:
        """Calculate polygon area in acres"""
        
        if len(coordinates) < 3:
            return None
        
        try:
            # Use shoelace formula for polygon area
            coords = [(c['latitude'], c['longitude']) for c in coordinates]
            
            # Convert to projected coordinates (simplified)
            x_coords = [c[1] for c in coords]  # longitude
            y_coords = [c[0] for c in coords]  # latitude
            
            area = 0.0
            n = len(coords)
            
            for i in range(n):
                j = (i + 1) % n
                area += x_coords[i] * y_coords[j]
                area -= x_coords[j] * y_coords[i]
            
            area = abs(area) / 2.0
            
            # Convert to acres (very rough approximation)
            # This would need proper coordinate system conversion in production
            area_acres = area * 247.105  # Rough conversion factor
            
            return area_acres
            
        except Exception as e:
            logger.warning(f"Area calculation failed: {str(e)}")
            return None
    
    def _calculate_geo_confidence(self, location_data: Dict, scale_info: Dict,
                                validation_result: Dict) -> float:
        """Calculate confidence score for geo-referencing results"""
        
        confidence = 0.0
        
        # Location data quality
        if location_data:
            if location_data['accuracy'] == 'address_level':
                confidence += 0.3
            elif location_data['accuracy'] == 'section_level':
                confidence += 0.2
        
        # Scale information
        if scale_info.get('scale_ratio'):
            confidence += 0.2
        
        # Validation results
        confidence += validation_result.get('overall_confidence', 0.0)
        
        return min(1.0, confidence)
    
    def _create_failure_result(self, error_message: str) -> Dict:
        """Create standardized failure result"""
        
        return {
            'status': 'failed',
            'error': error_message,
            'calculated_vertices': [],
            'confidence_score': 0.0
        } 