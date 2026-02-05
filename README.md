# House Price Prediction API

A machine learning-powered API for predicting house prices based on property features. Built with FastAPI and scikit-learn, deployed on Render.

## Features

- **Machine Learning Model**: Trained on real estate data to predict house prices
- **RESTful API**: FastAPI-based endpoints for predictions
- **Batch Processing**: Support for multi-core batch predictions
- **Auto Documentation**: Swagger UI and ReDoc available at `/docs` and `/redoc`
- **Health Monitoring**: Health check endpoint for uptime monitoring

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd House_Pricwe
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

### Training the Model

Before running the API, train the model:

```bash
python ml/train_final.py
```

This will:
- Load and preprocess ~247K property records
- Apply outlier removal (1st-99th percentile)
- Engineer 31 features (room ratios, age indicators, location encoding, etc.)
- Train LightGBM and XGBoost with 5-fold cross-validation
- Select the best model based on R² score
- Save the model to `artifacts/model.joblib`
- Write training logs to `logs/training_YYYYMMDD_HHMMSS.log`

### Running the API

Start the development server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/api/v1/health` | GET | Health check with model status |
| `/api/v1/predict` | POST | Predict single property price |
| `/api/v1/predict/batch` | POST | Predict multiple property prices |
| `/api/v1/model/info` | GET | Get model information |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

## API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Downtown",
    "size": 1500,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 2010,
    "condition": "Good",
    "property_type": "Single Family"
  }'
```

Response:
```json
{
  "predicted_price": 350000.0,
  "currency": "USD",
  "model_name": "XGBoost"
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "properties": [
      {
        "location": "Downtown",
        "size": 1500,
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 2010,
        "condition": "Good",
        "property_type": "Single Family"
      },
      {
        "location": "Suburb",
        "size": 2000,
        "bedrooms": 4,
        "bathrooms": 3,
        "year_built": 2015,
        "condition": "Excellent",
        "property_type": "Single Family"
      }
    ]
  }'
```

## Project Structure

```
House_Pricwe/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   └── routes.py        # API endpoints
│   ├── core/
│   │   └── config.py        # Configuration
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   └── services/
│       └── predictor.py     # Prediction service
├── ml/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_final.py       # Model training (high-performance)
│   └── evaluate.py          # Model evaluation
├── notebooks/
│   └── EDA.ipynb            # Exploratory analysis
├── artifacts/
│   └── model.joblib         # Trained model
├── logs/
│   └── training_*.log       # Training logs
├── tests/
│   └── test_api.py          # API tests
├── requirements.txt
├── Dockerfile
├── render.yaml
└── README.md
```

## Deployment on Render

### Option 1: Using render.yaml (Recommended)

1. Push your code to GitHub
2. Connect your repository to Render
3. Render will automatically detect `render.yaml` and deploy

### Option 2: Manual Setup

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`

### Keep-Alive CRON (Free Tier)

Render's free tier spins down after 15 minutes of inactivity. To keep the server awake:

1. Go to [cron-job.org](https://cron-job.org) (free)
2. Create an account
3. Add a new cron job:
   - **URL**: `https://your-app.onrender.com/health`
   - **Execution schedule**: Every 14 minutes
   - **Method**: GET

Alternative: Use [UptimeRobot](https://uptimerobot.com) with a 5-minute check interval.

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
# Install dev dependencies
pip install black isort

# Format code
black .
isort .
```

### Running EDA Notebook

```bash
jupyter notebook notebooks/EDA.ipynb
```

## Model Information

### Current Model Performance (LightGBM)

| Metric | Value |
|--------|-------|
| Test R² | **0.9816** |
| Test MAE | **$18,983** |
| Test RMSE | **$28,430** |
| Test MAPE | **4.83%** |

### Training Pipeline

1. **Data Processing** (247,172 → 232,275 records):
   - Outlier removal (1st-99th percentile)
   - Median imputation for numerical, mode for categorical

2. **Feature Engineering** (31 features):
   - **Date Features**: sale_year, sale_month, sale_quarter, cyclical encoding (month_sin, month_cos)
   - **Property Features**: property_age, is_new (≤5yr), is_old (>50yr)
   - **Room Ratios**: size_per_bedroom, size_per_bathroom, bed_bath_ratio, total_rooms
   - **Location Encoding**: loc_price_mean, loc_price_median, loc_count (target encoding)
   - **Interactions**: size_x_condition, age_x_condition

3. **Preprocessing**:
   - Numerical: RobustScaler (resistant to outliers)
   - Categorical: OneHotEncoder with min_frequency=0.5%

4. **Target Transformation**: log1p(price) → expm1(prediction)

5. **Model Selection**: LightGBM with 5-fold cross-validation

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug mode |
| `PORT` | `8000` | Server port |
| `MODEL_PATH` | `artifacts/model.joblib` | Path to model file |

## License

This project is for educational purposes as part of a data science case study.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
