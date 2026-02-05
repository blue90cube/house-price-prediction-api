# House Price Prediction - Case Study Presentation

---

## Slide 1: Project Overview

### Objective
Develop a machine learning model to predict house prices based on real estate property features and deploy it as a production-ready API.

### Deliverables
- Exploratory Data Analysis (EDA)
- ML Model with multi-core processing support
- RESTful API for predictions
- Deployment configuration for Render

---

## Slide 2: Dataset Overview

### Data Source
- **File**: Case Study 1 Data.xlsx
- **Records**: 247,172 properties (232,275 after outlier removal)
- **Features**: 10 columns → 31 engineered features

### Features
| Feature | Type | Description |
|---------|------|-------------|
| Property ID | Identifier | Unique property identifier |
| Location | Categorical | City/Neighborhood |
| Size | Numerical | Square feet |
| Bedrooms | Numerical | Count |
| Bathrooms | Numerical | Count |
| Year Built | Numerical | Construction year |
| Condition | Categorical | Property condition |
| Type | Categorical | Property type |
| Date Sold | Date | Sale date |
| **Price** | **Target** | **Sale price** |

---

## Slide 3: Exploratory Data Analysis

### Key EDA Steps
1. **Data Loading & Inspection**
   - Verified data types
   - Checked dataset dimensions

2. **Missing Value Analysis**
   - Identified missing patterns
   - Applied appropriate imputation strategies

3. **Duplicate Detection**
   - Removed duplicate Property IDs

4. **Distribution Analysis**
   - Price distribution (right-skewed)
   - Numerical feature distributions
   - Categorical value counts

---

## Slide 4: EDA Findings - Price Distribution

### Target Variable Characteristics
- **Skewness**: Right-skewed distribution
- **Outliers**: Present in upper price range
- **Range**: Wide variation in property prices

### Key Insights
- Log transformation may improve model performance
- Outlier handling required
- Price correlates strongly with Size and Location

---

## Slide 5: Feature Engineering

### Date Features (6)
- `sale_year`, `sale_month`, `sale_quarter`
- Cyclical encoding: `month_sin`, `month_cos`

### Property Features (7)
- `property_age` = sale_year - year_built
- `is_new` (≤5 years), `is_old` (>50 years)
- `log_size` = log(size + 1)

### Room-Based Ratios (5)
- `size_per_bedroom`, `size_per_bathroom`
- `bed_bath_ratio`, `total_rooms`, `size_per_room`

### Target Encoding (5)
- `loc_price_mean`, `loc_price_median`, `loc_count` (by location)
- `type_price_mean`, `condition_num`

### Interaction Features (2)
- `size_x_condition`, `age_x_condition`

---

## Slide 6: Preprocessing Pipeline

### Numerical Features (25)
```
RobustScaler (median-based, outlier resistant)
```
Features: size, bedrooms, bathrooms, year_built, property_age, room ratios, log_size, cyclical encodings, target encodings, interaction features

### Categorical Features (3)
```
OneHotEncoder (handle_unknown='ignore', min_frequency=0.5%)
```
Features: Location, Condition, Type

### Pipeline Architecture
```
Raw Data → Outlier Removal → Imputation → Feature Engineering → ColumnTransformer → TransformedTargetRegressor(log1p) → LightGBM
```

---

## Slide 7: Model Selection

### Models Evaluated
| Model | Description | Multi-core |
|-------|-------------|------------|
| Linear Regression | Baseline | No |
| Ridge/Lasso | Regularized linear | No |
| Random Forest | Ensemble trees | Yes |
| Gradient Boosting | Sequential boosting | No |
| XGBoost | Optimized GBM | Yes |
| LightGBM | Fast gradient boosting | Yes |

### Evaluation Metrics
- **MAE** (Mean Absolute Error) - Primary metric
- **RMSE** (Root Mean Square Error)
- **R²** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

---

## Slide 8: Multi-Core Processing

### Training Phase
```python
# Cross-validation with parallel processing
cross_val_score(model, X, y, cv=5, n_jobs=-1)

# Hyperparameter tuning
RandomizedSearchCV(model, param_grid, n_jobs=-1)
```

### Inference Phase
```python
# Batch predictions with ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    predictions = list(executor.map(predict_single, properties))
```

### Benefits
- Faster training times
- Efficient batch predictions
- Better resource utilization

---

## Slide 9: Model Results

### Final Model Performance (LightGBM)

| Metric | Cross-Validation | Test Set |
|--------|------------------|----------|
| **R²** | 0.9816 | **0.9816** |
| **MAE** | $19,018 | **$18,983** |
| **RMSE** | - | **$28,430** |
| **MAPE** | 4.84% | **4.83%** |

### Model Comparison (5-Fold CV)
| Model | CV R² | CV MAE | CV MAPE |
|-------|-------|--------|---------|
| **LightGBM** | **0.9816** | **$19,018** | **4.84%** |
| XGBoost | 0.9797 | $20,089 | 5.11% |

### Best Model: LightGBM
Selected based on highest R² score with:
- 500 estimators, max_depth=15, learning_rate=0.03
- Log target transformation (log1p/expm1)
- RobustScaler for numerical features

---

## Slide 10: API Architecture

### Technology Stack
- **Framework**: FastAPI
- **Server**: Uvicorn (ASGI)
- **Validation**: Pydantic
- **Documentation**: OpenAPI (Swagger/ReDoc)

### Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/api/v1/predict` | POST | Single prediction |
| `/api/v1/predict/batch` | POST | Batch predictions |
| `/api/v1/model/info` | GET | Model information |

---

## Slide 11: API Request/Response

### Single Prediction Request
```json
POST /api/v1/predict
{
  "location": "Downtown",
  "size": 1500,
  "bedrooms": 3,
  "bathrooms": 2,
  "year_built": 2010,
  "condition": "Good",
  "property_type": "Single Family"
}
```

### Response
```json
{
  "predicted_price": 350000.00,
  "currency": "USD",
  "model_name": "XGBoost"
}
```

---

## Slide 12: Deployment Architecture

```
                    ┌──────────────────┐
                    │    Client App    │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Render (Free)   │
                    │  ┌─────────────┐ │
                    │  │   FastAPI   │ │
                    │  │   Server    │ │
                    │  └──────┬──────┘ │
                    │         │        │
                    │  ┌──────▼──────┐ │
                    │  │  ML Model   │ │
                    │  │  Pipeline   │ │
                    │  └─────────────┘ │
                    └──────────────────┘
                             ▲
            ┌────────────────┴────────────────┐
            │         cron-job.org            │
            │   (ping /health every 14 min)   │
            └─────────────────────────────────┘
```

---

## Slide 13: Keep-Alive Strategy

### Challenge
Render free tier spins down after 15 minutes of inactivity.

### Solution
External CRON service to keep the server awake.

### Implementation
1. **cron-job.org** (Recommended)
   - Free service
   - Ping `https://your-app.onrender.com/health` every 14 minutes

2. **UptimeRobot** (Alternative)
   - Free monitoring
   - 5-minute check intervals

### Health Endpoint
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": True
    }
```

---

## Slide 14: Project Structure

```
House_Pricwe/
├── app/                    # FastAPI application
│   ├── main.py            # Application entry
│   ├── api/routes.py      # API endpoints
│   ├── core/config.py     # Configuration
│   ├── models/schemas.py  # Pydantic models
│   └── services/          # Business logic
├── ml/                    # Machine learning
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation
│   ├── data_preprocessing.py
│   └── feature_engineering.py
├── notebooks/EDA.ipynb   # Analysis
├── artifacts/            # Model files
├── tests/                # Unit tests
├── requirements.txt
├── Dockerfile
└── render.yaml           # Deployment config
```

---

## Slide 15: Running the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run EDA
```bash
jupyter notebook notebooks/EDA.ipynb
```

### 3. Train Model
```bash
python ml/train_final.py
```
Outputs: `artifacts/model.joblib` + `logs/training_*.log`

### 4. Start API
```bash
uvicorn app.main:app --reload
```

### 5. Access Documentation
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Slide 16: Key Technical Decisions

### 1. Model Serialization
**Choice**: `joblib` over `pickle`
- Better numpy array handling
- Faster serialization for sklearn objects

### 2. Feature Engineering in Pipeline
**Approach**: Preprocessing included in pipeline
- Ensures consistency between training and inference
- Handles unseen categories gracefully

### 3. API Framework
**Choice**: FastAPI
- Automatic documentation
- Built-in validation
- Async support
- High performance

### 4. Deployment Platform
**Choice**: Render (free tier)
- Simple deployment from Git
- Automatic HTTPS
- Environment variable support

---

## Slide 17: Future Improvements

### Model Enhancements
- [ ] Feature selection optimization
- [ ] Ensemble of multiple models
- [ ] Neural network approach
- [ ] AutoML integration

### API Improvements
- [ ] Caching for repeated predictions
- [ ] Rate limiting
- [ ] Authentication
- [ ] Prediction confidence intervals

### DevOps
- [ ] CI/CD pipeline
- [ ] Monitoring and logging
- [ ] A/B testing for model versions
- [ ] Model versioning

---

## Slide 18: Conclusion

### Achievements
- **High-Performance Model**: R²=0.9816, MAE=$18,983, MAPE=4.83%
- **Production-Ready API**: FastAPI with batch processing support
- **Comprehensive Feature Engineering**: 31 features from 10 raw columns
- **Proper Logging**: Training logs saved for reproducibility

### Key Learnings
- End-to-end ML system development
- Advanced feature engineering techniques
- Target transformation for skewed distributions
- Cloud deployment strategies (Render free tier)

### API Link
`https://your-app.onrender.com`

---

## Slide 19: Q&A

### Resources
- **GitHub**: [Repository URL]
- **API Docs**: /docs
- **Model Metrics**: Run `python -m ml.train`

### Contact
For questions or feedback, please reach out.

---

## Appendix A: Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov=ml

# Run specific test file
pytest tests/test_api.py -v
```

---

## Appendix B: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEBUG` | `false` | Enable debug mode |
| `PORT` | `8000` | Server port |
| `MODEL_PATH` | `artifacts/model.joblib` | Model file path |
| `MAX_BATCH_SIZE` | `100` | Max batch size |
| `BATCH_WORKERS` | `4` | Parallel workers |

---

## Appendix C: API Error Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request |
| 422 | Validation Error |
| 500 | Internal Server Error |
| 503 | Service Unavailable (Model not loaded) |
