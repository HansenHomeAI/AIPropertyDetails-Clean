# AIPropertyDetails

An AI-powered system for analyzing parcel maps and legal documentation to extract property details, with a primary focus on determining geocoordinates of parcel boundaries for 3D modeling applications.

## Overview

This system uses OpenAI's o4-mini model (released in 2025) for advanced visual reasoning to analyze various types of property documents including:

- Parcel maps
- Plat documents  
- Legal property descriptions
- Survey documents
- Tax assessor maps

The primary goal is to extract precise geocoordinates that define property boundaries, enabling accurate replication in 3D models.

## Key Features

- **Advanced Visual Analysis**: Leverages o4-mini's cutting-edge visual reasoning capabilities
- **Boundary Coordinate Extraction**: Primary focus on determining precise parcel boundary coordinates
- **Multi-Document Support**: Handles various types of property documentation
- **Address & Location Detection**: Extracts property addresses and location information
- **Robust Processing**: Designed for reliable analysis across diverse document formats
- **Local Development**: Flask-based local server for development and testing

## Technology Stack

- **Backend**: Flask (Python)
- **AI Model**: OpenAI o4-mini (via API)
- **Document Processing**: Advanced image analysis and OCR capabilities
- **Geospatial**: Coordinate extraction and validation
- **Development**: Local development environment with plans for AWS production deployment

## Project Structure

```
AIPropertyDetails/
├── app/
│   ├── __init__.py
│   ├── routes.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── property_analyzer.py
│   │   └── coordinate_extractor.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── openai_service.py
│   │   └── document_processor.py
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py
│       └── validation.py
├── tests/
├── static/
├── templates/
├── uploads/
├── requirements.txt
├── config.py
├── run.py
└── README.md
```

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AIPropertyDetails.git
cd AIPropertyDetails
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export OPENAI_API_KEY="your-o4-mini-api-key-here"
```

5. Run the application:
```bash
python run.py
```

## Usage

1. Start the Flask server
2. Navigate to `http://localhost:5000`
3. Upload a parcel map or property document
4. Review the extracted boundary coordinates and property details
5. Export results for use in 3D modeling applications

## API Endpoints

- `POST /api/analyze` - Upload and analyze property documents
- `GET /api/results/{id}` - Retrieve analysis results
- `POST /api/validate` - Validate extracted coordinates

## Development Roadmap

### Phase 1: Core Development (Current)
- ✅ Project setup and repository initialization
- 🔄 Basic Flask application structure
- 🔄 OpenAI o4-mini integration
- 🔄 Document upload and processing
- 🔄 Basic coordinate extraction

### Phase 2: Enhanced Analysis
- 📋 Advanced boundary detection algorithms
- 📋 Multi-document correlation analysis
- 📋 Coordinate validation and verification
- 📋 Address and location enrichment

### Phase 3: Production Ready
- 📋 AWS deployment architecture
- 📋 Scalable processing pipeline
- 📋 API rate limiting and optimization
- 📋 Comprehensive testing suite

### Phase 4: Advanced Features
- 📋 Batch processing capabilities
- 📋 Integration with GIS systems
- 📋 Historical property data analysis
- 📋 Machine learning model improvements

## Contributing

This project is currently under active development. Contributions and suggestions are welcome!

## License

[License details to be added]

---

**Note**: This system is designed to assist with property analysis but should not be used as the sole source for legal or official property boundary determinations. Always consult with licensed surveyors and legal professionals for official property information. 