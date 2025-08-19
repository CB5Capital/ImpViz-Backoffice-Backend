# ImpViz Trading Analytics API Backend

Professional FastAPI backend for trading analytics and regime analysis.

## 🚀 Quick Start

### Installation
```bash
cd /Users/viktorkardvik/Code/CB5Capital/ImpViz/API_Backend
pip install -r requirements.txt
```

### Configuration
1. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your database credentials
```

### Run the API
```bash
uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
```

### API Documentation
Once running, visit:
- **Interactive docs**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## 📡 API Endpoints

### Health & Status
- `GET /` - API status
- `GET /health` - Database health check

### Regime Analysis
- `GET /regime-models` - Get all regime models
- `GET /current-regime/{model_id}` - Get current regime
- `GET /regime-data/{model_id}` - Get regime data and counts

### Target Metrics
- `GET /target-metrics` - Get all target metrics
- `GET /engineered-metrics` - Get engineered metric names
- `GET /target-metric-categories` - Get metric categories
- `POST /target-metrics` - Create new target metric

### Analysis Tools
- `POST /correlation-analysis` - Perform correlation analysis
- `POST /setup-list-data` - Get filtered setup data
- `GET /dropdown-data` - Get all dropdown options

### Utilities
- `POST /run-routine-script` - Execute routine operations
- `GET /feature-engineering-functions` - Get available functions

## 🗃️ Database

The API connects to PostgreSQL database with the following tables:
- `regime_model`, `regime_row`
- `target_metric`, `target_metric_row`, `target_metric_metadata_row`
- `engineered_target_metric_row`
- `feature`, `feature_row`
- And more...

## 🔧 Development

### Project Structure
```
API_Backend/
├── api_server.py              # Main FastAPI application
├── database.py                # Database connection utilities
├── target_metrics.py          # Trading metrics business logic
├── feature_engineering_functions.py
├── regime_model.py            # Regime analysis logic
├── routine_scripts.py         # Maintenance scripts
├── feature_bulk_uploader.py   # Data ingestion
├── feature_updater.py         # Data updates
├── requirements.txt           # Dependencies
├── .env                       # Environment configuration
└── README.md                  # This file
```

### Environment Variables (.env)
```bash
DB_USERNAME=your_username
DB_PASSWORD=your_password  
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database
DB_SSLMODE=require
```

## 🔍 Testing

Test the API endpoints:
```bash
# Health check
curl http://127.0.0.1:8000/health

# Get regime models
curl http://127.0.0.1:8000/regime-models

# Get dropdown data
curl http://127.0.0.1:8000/dropdown-data
```

## 📈 Performance

- **Async endpoints** for non-blocking operations
- **Optimized database queries** with batch processing
- **Connection pooling** via SQLAlchemy
- **Background task processing** for heavy computations

## 🐛 Troubleshooting

### Common Issues
1. **Database connection failed**: Check .env configuration
2. **Port already in use**: Kill existing uvicorn processes or use different port
3. **Import errors**: Ensure all dependencies are installed

### Logs
API logs are displayed in the terminal where uvicorn is running.

---

**API Backend Ready! 🚀**