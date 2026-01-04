# Synthetic Review Data Generator

A powerful tool for generating synthetic product reviews using Large Language Models (LLMs). Built with LangChain and LangGraph, this generator creates realistic, diverse reviews while maintaining quality through semantic similarity checks, sentiment alignment, and reality validation.

## Features

- **Multi-Model Support**: Works with OpenAI, Ollama, and Mistral AI models
- **Persona-Based Generation**: Generate reviews from different user perspectives (CEO, Trainer, L&D Manager, etc.)
- **Quality Assurance**: Built-in checks for semantic similarity, sentiment alignment, and reality validation
- **Research Integration**: Optional web research to ground reviews in real product information
- **Comprehensive Reporting**: Quality reports, comparison metrics, and performance summaries
- **REST API**: FastAPI-based API for integration with other services
- **Docker Support**: Easy deployment with Docker

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
  - [CLI](#cli)
  - [REST API](#rest-api)
- [Docker](#docker)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- API keys for your chosen LLM provider(s)

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Synthetic Data Generator"
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Or using conda
   conda create -n synthetic-generator python=3.11
   conda activate synthetic-generator
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   # For OpenAI
   OPENAI_API_KEY=your-openai-api-key
   
   # For Mistral AI (optional)
   MISTRAL_API_KEY=your-mistral-api-key
   
   # For Ollama (optional) - no API key needed, just run Ollama locally
   ```

5. **Verify installation**
   ```bash
   python -c "from modules.graph import build_graph; print('Installation successful!')"
   ```

## Quick Start

1. **Using default configuration**
   ```bash
   python app.py
   ```

2. **Using custom configuration file**
   ```bash
   python app.py --config my_config.json
   ```

3. **Using JSON string configuration**
   ```bash
   python app.py --config '{"products": [{"name": "MyProduct", "type": "SaaS", "description": "A great product"}], "personas": [{"role": "User", "description": "Regular user"}], "rating_distribution": {"1": 0.1, "2": 0.1, "3": 0.2, "4": 0.3, "5": 0.3}, "models": [{"provider": "openai", "name": "gpt-4o-mini"}], "samples_number": 5, "similarity_threshold": 0.7, "options": {"pros_and_cons": true, "rating": true, "use_research": false}}'
   ```

## Configuration

The configuration file (`config.json`) defines the generation parameters:

```json
{
    "products": [
        {
            "name": "Easygenerator",
            "type": "E-learning platform",
            "description": "Cloud-based e-learning platform for creating online courses."
        }
    ],
    "personas": [
        {
            "role": "Instructional Designer",
            "description": "Focuses on course creation efficiencies and learner engagement."
        },
        {
            "role": "CEO",
            "description": "Focuses on business value, ROI, and scalability."
        }
    ],
    "rating_distribution": {
        "1": 0.1,
        "2": 0.1,
        "3": 0.2,
        "4": 0.3,
        "5": 0.3
    },
    "models": [
        {
            "provider": "openai",
            "name": "gpt-4o-mini"
        }
    ],
    "samples_number": 10,
    "similarity_threshold": 0.7,
    "options": {
        "pros_and_cons": true,
        "rating": true,
        "use_research": true
    }
}
```

### Configuration Options

| Field | Type | Description |
|-------|------|-------------|
| `products` | array | List of products to generate reviews for |
| `personas` | array | List of reviewer personas |
| `rating_distribution` | object | Probability distribution for ratings (1-5) |
| `models` | array | LLM configurations (provider + name) |
| `samples_number` | integer | Number of reviews to generate |
| `similarity_threshold` | float | Max similarity allowed between reviews (0-1) |
| `options.pros_and_cons` | boolean | Include pros/cons in reviews |
| `options.rating` | boolean | Include ratings in reviews |
| `options.use_research` | boolean | Enable web research for context |

### Supported Model Providers

| Provider | Example Models | Notes |
|----------|---------------|-------|
| `openai` | `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo` | Requires `OPENAI_API_KEY` |
| `ollama` | `llama3.1`, `mistral`, `mixtral` | Run Ollama locally |
| `mistral` | `mistral-large-latest` | Requires `MISTRAL_API_KEY` |

## Usage

### CLI

```bash
# Basic usage with default config
python app.py

# Custom config file
python app.py --config path/to/config.json

# JSON string config (useful for scripts/automation)
python app.py -c '{"products": [...], ...}'

# Help
python app.py --help
```

### REST API

Start the API server:

```bash
# Using Python directly
python api.py

# Using uvicorn (recommended for production)
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. View the interactive documentation at `http://localhost:8000/docs`.

#### Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/generate` | POST | Start async generation job |
| `/generate/sync` | POST | Run generation synchronously |
| `/generate/default` | GET | Start generation with default config |
| `/jobs` | GET | List all generation jobs |
| `/jobs/{job_id}` | GET | Get job status |
| `/outputs` | GET | List generated output files |
| `/outputs/{filename}` | GET | Download output file |
| `/reports` | GET | List generated reports |
| `/config/default` | GET | Get default configuration |

#### Example API Usage

```bash
# Start a generation job
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "products": [{"name": "MyApp", "type": "SaaS", "description": "A productivity app"}],
    "personas": [{"role": "User", "description": "Regular user"}],
    "models": [{"provider": "openai", "name": "gpt-4o-mini"}],
    "samples_number": 5
  }'

# Check job status
curl "http://localhost:8000/jobs/{job_id}"

# List outputs
curl "http://localhost:8000/outputs"

# Download output
curl "http://localhost:8000/outputs/reviews_gpt-4o-mini_20260104_120000.json"
```

## Docker

### Building the Image

```bash
docker build -t synthetic-data-generator .
```

### Running with Docker

**Run the API server:**
```bash
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-api-key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/reports:/app/reports \
  synthetic-data-generator
```

**Run CLI mode:**
```bash
docker run --rm \
  -e OPENAI_API_KEY=your-api-key \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/config.json:/app/config.json \
  synthetic-data-generator python app.py --config config.json
```

**Run with custom config JSON:**
```bash
docker run --rm \
  -e OPENAI_API_KEY=your-api-key \
  -v $(pwd)/data:/app/data \
  synthetic-data-generator python app.py --config '{"products": [...], ...}'
```

### Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  generator:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./data:/app/data
      - ./reports:/app/reports
      - ./config.json:/app/config.json
    restart: unless-stopped
```

Run with:
```bash
docker-compose up -d
```

## Project Structure

```
Synthetic Data Generator/
├── app.py                 # CLI application
├── api.py                 # FastAPI REST API
├── config.json            # Default configuration
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this)
├── data/
│   ├── real_reviews.json  # Reference reviews for comparison
│   └── output/            # Generated reviews
├── modules/
│   ├── embedder.py        # Embedding utilities
│   ├── graph.py           # LangGraph workflow
│   ├── metrics.py         # Performance tracking
│   ├── prompts.py         # LLM prompts
│   ├── QA.py              # Quality assurance checks
│   ├── reporting.py       # Report generation
│   ├── research.py        # Web research module
│   └── utils.py           # Utility functions
├── reports/               # Generated quality reports
└── tests/                 # Test files
```

## Output Format

Generated reviews are saved as JSON with the following structure:

```json
[
    {
        "title": "Excellent e-learning platform",
        "reviewer_role": "Instructional Designer",
        "comment": "The platform has transformed how we create courses...",
        "pros": "Easy to use, great templates, excellent support",
        "cons": "Some advanced features require learning curve",
        "rating": 5,
        "product_name": "Easygenerator"
    }
]
```

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **API key errors**: Verify your `.env` file contains the correct API keys

3. **Ollama connection errors**: Ensure Ollama is running locally:
   ```bash
   ollama serve
   ```

4. **Memory issues**: For large batch sizes, consider reducing `samples_number` or using a smaller embedding model

### Logs

Generation logs are saved to `reports/last_run.log` for debugging.
