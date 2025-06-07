"""
Property Database Service - Searches government databases for property coordinates
Now uses dynamic database discovery instead of hardcoded APIs
"""

import logging
from typing import Dict, List, Optional
from .dynamic_database_service import DynamicDatabaseService

logger = logging.getLogger(__name__)

class PropertyDatabaseService:
    """Service for searching government property databases dynamically"""
    
    def __init__(self, openai_service):
        self.openai_service = openai_service
        self.dynamic_service = DynamicDatabaseService(openai_service)
    
    def search_all_databases(self, property_details: Dict) -> Dict:
        """
        Search government databases for property coordinates
        Uses dynamic discovery to find relevant databases for any location
        """
        
        logger.info("Starting comprehensive database search using dynamic discovery")
        
        # Use the dynamic service to discover and search databases
        result = self.dynamic_service.discover_and_search_databases(property_details)
        
        if result['coordinates_found']:
            logger.info(f"SUCCESS: Found coordinates from {result['source']} with {result['confidence']*100:.1f}% confidence")
            return self._format_database_result(result)
        else:
            logger.warning("No coordinates found in any government database")
            return {
                'vertices': [],
                'confidence': 0.0,
                'source': 'government_databases',
                'method': 'dynamic_search',
                'details': {
                    'databases_searched': len(result.get('discovered_databases', [])),
                    'discovered_databases': result.get('discovered_databases', [])
                }
            }
    
    def _format_database_result(self, result: Dict) -> Dict:
        """Format the result from dynamic database search"""
        
        return {
            'vertices': result['vertices'],
            'confidence': result['confidence'],
            'source': result['source'],
            'method': 'dynamic_database_search',
            'details': {
                'databases_searched': len(result.get('discovered_databases', [])),
                'successful_database': result['source'],
                'search_results': result.get('search_results', {}),
                'discovered_databases': result.get('discovered_databases', [])
            }
        } 