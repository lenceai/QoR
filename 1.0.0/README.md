# CERN Knowledge Explorer v1.0.0

A comprehensive system for exploring, analyzing, and visualizing scientific data and research from CERN. The system provides advanced search capabilities, data visualization, and knowledge discovery tools for researchers and scientists.

## Features

- **Advanced Search**: Multi-field search with Boolean operations, fuzzy matching, and faceted filtering
- **Knowledge Discovery**: Topic modeling, citation analysis, and research trend identification
- **Data Visualization**: Interactive charts, network graphs, and timeline visualizations
- **Research Analytics**: Publication metrics, collaboration mapping, and impact analysis
- **Modern Web Interface**: Responsive design with real-time updates
- **API Access**: RESTful API for programmatic access to data and functionality

## Quick Start

### Prerequisites

- Python 3.11 or higher
- PostgreSQL 12 or higher
- Elasticsearch 8.0 or higher
- Redis 6.0 or higher

### Installation

1. Clone the repository and navigate to the project directory:
```bash
cd 1.0.0
```

2. Create a virtual environment:
```bash
conda create -n QoR python=3.11
conda activate QoR
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database:
```bash
python src/scripts/init_db.py
```

6. Start the application:
```bash
python src/main.py
```

The application will be available at `http://localhost:8000`

## Project Structure

```
1.0.0/
├── data/                   # Input data and configurations
│   ├── config/            # Configuration files
│   ├── samples/           # Sample datasets
│   └── schemas/           # Database schemas
├── output/                # Generated reports and logs
│   ├── logs/             # Application logs
│   ├── reports/          # Generated reports
│   └── exports/          # Data exports
├── src/                   # Source code
│   ├── api/              # API endpoints
│   ├── core/             # Core functionality
│   ├── data/             # Data models and access
│   ├── services/         # Business logic
│   └── utils/            # Utility functions
├── tests/                 # Test files
├── docs/                 # Documentation
└── scripts/              # Utility scripts
```

## Phase-by-Phase Implementation

This project is developed in 8 phases:

1. **Phase 1**: Foundation and Core Infrastructure ✅
2. **Phase 2**: Data Ingestion and Processing
3. **Phase 3**: Search and Discovery Engine
4. **Phase 4**: Visualization and Analytics
5. **Phase 5**: Web Interface and User Experience
6. **Phase 6**: Integration and Advanced Features
7. **Phase 7**: Performance and Scalability
8. **Phase 8**: Testing, Documentation, and Deployment

## API Documentation

Once the application is running, visit:
- API Documentation: `http://localhost:8000/docs`
- Alternative Documentation: `http://localhost:8000/redoc`

## Configuration

Key configuration options in `.env`:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/cern_explorer

# Elasticsearch
ELASTICSEARCH_URL=http://localhost:9200

# Redis
REDIS_URL=redis://localhost:6379

# Application
DEBUG=true
SECRET_KEY=your-secret-key
API_V1_STR=/api/v1
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
isort src/
```

### Type Checking

```bash
mypy src/
```

## Contributing

1. Follow the established code style (Black, isort)
2. Write tests for new functionality
3. Update documentation as needed
4. Submit pull requests for review

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CERN for providing the data and research inspiration
- The scientific community for their valuable contributions
- Open source libraries that make this project possible

## Support

For support and questions:
- Documentation: `/docs`
- Issues: Create an issue in the repository
- Email: support@cern-explorer.org 