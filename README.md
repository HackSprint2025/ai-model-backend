# AI Model Backend

A FastAPI-based REST API for disease prediction using a machine learning model with 24 input features.

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── predict.py          # Model loading and prediction logic
│   └── schemas.py          # Pydantic models for request/response validation
├── model/
│   └── disease prediction model.pkl  # Trained ML model
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
└── README.md
```

## Features

- FastAPI-based REST API
- CORS support for cross-origin requests
- Machine learning model for disease prediction
- 24-feature input validation
- Confidence score for predictions
- Hot reload support for development

## API Endpoints

### GET /
Returns API information and usage instructions.

### POST /predict
Accepts 24 float features and returns a prediction with confidence score.

Request body:
```json
{
  "features": [float1, float2, ..., float24]
}
```

Response:
```json
{
  "prediction": "predicted_class",
  "confidence": 0.95
}
```

## Requirements

- Python 3.11 or higher
- pip (Python package installer)
- Git

## Installation and Setup

### Clone the Repository

```bash
git clone https://github.com/HackSprint2025/ai-model-backend.git
cd ai-model-backend
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

On Windows:
```bash
venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

### Start the Development Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- Local: http://localhost:8000
- Network: http://0.0.0.0:8000

### Access API Documentation

Once the server is running, access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Technology Stack

- FastAPI: Web framework for building APIs
- Uvicorn: ASGI server
- Pydantic: Data validation using Python type annotations
- NumPy: Numerical computing
- Scikit-learn: Machine learning library
- Pickle: Model serialization

## Development

The application uses FastAPI's hot reload feature. Any changes to the Python files will automatically restart the server during development.

## CORS Configuration

The API is configured to accept requests from all origins. For production deployment, update the FRONTEND_ORIGINS variable in main.py to specify allowed origins.
