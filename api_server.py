from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import uvicorn
from datetime import datetime
import asyncio
import logging
import pandas as pd
import numpy as np
import time
from scipy import stats

# Import existing modules
from database import get_db, close_db, insert_into_table, get_session
from target_metrics import (
    get_live_regime_data, get_regime_data, get_historical_regime_data, get_current_regime, 
    get_batch_setup_list_data, get_mean_metric_per_regime,
    insert_new_target_metric, run_backtest_function
)
from routine_scripts import main as routine_script
from feature_engineering_functions import export_functions, function_requirements
from feature_bulk_uploader import market_data, macro_data
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# UTILITY FUNCTIONS FOR NUMPY TYPE CONVERSION
# =============================================================================

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.
    Handles nested dictionaries, lists, and all numpy scalar and array types.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.datetime64, pd.Timestamp)):
        if pd.isna(obj):
            return None
        return obj.isoformat() if hasattr(obj, 'isoformat') else str(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

# =============================================================================
# CACHING SYSTEM
# =============================================================================

class SimpleCache:
    """Simple in-memory cache with TTL (Time To Live)"""
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str, ttl_seconds: int = 300) -> Optional[Any]:
        """Get cached value if not expired (default TTL: 5 minutes)"""
        if key not in self._cache:
            return None
        
        # Check if expired
        if time.time() - self._timestamps[key] > ttl_seconds:
            self.delete(key)
            return None
        
        return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value with current timestamp"""
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def delete(self, key: str) -> None:
        """Delete cached value"""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cached values"""
        self._cache.clear()
        self._timestamps.clear()

# Global cache instance
cache = SimpleCache()

# Initialize FastAPI
app = FastAPI(
    title="ImpViz Trading Analytics API",
    description="Professional Trading Analytics Platform Backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://0.0.0.0:3000"],  # Allow various localhost formats
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# PYDANTIC MODELS FOR REQUEST/RESPONSE
# =============================================================================

class DatabaseConnectionResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime

class RegimeModel(BaseModel):
    id: int
    name: str

class TargetMetric(BaseModel):
    id: int
    name: str
    sub_category: Optional[str]
    timeframe: Optional[str]

class CorrelationRequest(BaseModel):
    model_id: int
    engineered_metric_name: str
    selected_setup_name: str
    selected_sub_category: str
    selected_timeframe: str

class CorrelationResult(BaseModel):
    setup: str
    sub_category: str
    timeframe: str
    setup_display: str
    correlation: float

class SetupListRequest(BaseModel):
    model_id: int
    category_id: int
    selected_metrics: List[str]
    selected_timeframes: List[str]
    selected_sub_categories: List[str]

class NewTargetMetricRequest(BaseModel):
    filename: str
    source: str
    function: str
    target_metric_name: str
    sub_category: str


class MetricsResponse(BaseModel):
    setup_data: Dict[tuple, Dict[str, float]]
    regime_data: Dict[str, Any]

# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

def get_database():
    """Dependency to get database connection"""
    db = get_db()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        close_db(db)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_model=DatabaseConnectionResponse)
async def root():
    """Health check endpoint"""
    return DatabaseConnectionResponse(
        status="success",
        message="ImpViz Trading Analytics API is running",
        timestamp=datetime.now()
    )

@app.get("/health", response_model=DatabaseConnectionResponse)
async def health_check():
    """Database health check"""
    try:
        with get_session() as db:
            # Test database connection
            result = db.execute(text("SELECT 1")).fetchone()
            return DatabaseConnectionResponse(
                status="success",
                message="Database connection healthy",
                timestamp=datetime.now()
            )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unhealthy: {str(e)}")

# =============================================================================
# REGIME AND MODEL ENDPOINTS
# =============================================================================

@app.get("/regime-models", response_model=List[Dict[str, Any]])
async def get_regime_models():
    """Get all regime models"""
    try:
        with get_session() as db:
            result = db.execute(text("SELECT id, model_name FROM regime_model ORDER BY model_name;")).fetchall()
            return [{"id": row[0], "model_name": row[1]} for row in result]
    except Exception as e:
        logger.error(f"Error fetching regime models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/current-regime/{model_id}")
async def get_current_regime_endpoint(model_id: int):
    """Get current regime for a model"""
    try:
        regime_data = get_current_regime(model_id)
        return regime_data.to_dict('records')
    except Exception as e:
        logger.error(f"Error fetching current regime: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/regime-data/{model_id}")
async def get_regime_data_endpoint(model_id: int):
    """Get historical regime probability data for time series visualization"""
    try:
        regime_df, regime_counts = get_historical_regime_data(model_id)
        return {
            "regime_data": regime_df.to_dict('records'),
            "regime_counts": regime_counts
        }
    except Exception as e:
        logger.error(f"Error fetching regime data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# TARGET METRICS ENDPOINTS
# =============================================================================

@app.get("/target-metrics")
async def get_target_metrics():
    """Get all target metrics"""
    try:
        with get_session() as db:
            result = db.execute(text("""
                SELECT DISTINCT tm.id, tm.name, tm.sub_category, mdr.value as timeframe
                FROM target_metric tm
                JOIN target_metric_row tmr ON tm.id = tmr.target_metric_id
                JOIN target_metric_metadata_row mdr ON mdr.target_metric_row_id = tmr.id
                WHERE mdr.name = 'Timeframe'
                ORDER BY tm.name, tm.sub_category, mdr.value
            """)).fetchall()
            
            return [{
                "id": row[0],
                "name": row[1], 
                "sub_category": row[2],
                "timeframe": row[3]
            } for row in result]
    except Exception as e:
        logger.error(f"Error fetching target metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/engineered-metrics")
async def get_engineered_metrics():
    """Get all engineered metric names"""
    try:
        with get_session() as db:
            result = db.execute(text("""
                SELECT DISTINCT name FROM engineered_target_metric_row 
                ORDER BY name
            """)).fetchall()
            return [row[0] for row in result]
    except Exception as e:
        logger.error(f"Error fetching engineered metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/target-metric-categories")
async def get_target_metric_categories():
    """Get all target metric categories"""
    try:
        with get_session() as db:
            result = db.execute(text("SELECT id, name FROM target_metric_category;")).fetchall()
            categories = [{"id": row[0], "name": row[1]} for row in result]
            
            return categories
    except Exception as e:
        logger.error(f"Error fetching target metric categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/target-metrics")
async def create_target_metric(
    file: UploadFile = File(...),
    source: str = Form(...),
    function: str = Form(...),
    target_metric_name: str = Form(...),
    sub_category: str = Form(...)
):
    """Create a new target metric with file upload"""
    import tempfile
    import os
    
    try:
        # Create a temporary file to store the uploaded data
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or '')[1]) as temp_file:
            # Write uploaded file content to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_filename = temp_file.name
        
        try:
            # Use the temporary file path for processing
            result = insert_new_target_metric(
                None, temp_filename, source, 
                function, target_metric_name, 
                sub_category
            )
            return {"status": "success", "message": "Target metric created successfully"}
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error creating target metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# SETUP LIST AND BATCH DATA ENDPOINTS
# =============================================================================

@app.post("/setup-list-data")
async def get_setup_list_data(request: SetupListRequest):
    """Get setup list data with filters"""
    try:
        # Handle empty filter lists
        if not request.selected_sub_categories or not request.selected_timeframes:
            logger.info("Empty filter lists, returning empty data")
            return {"setup_data": {}}
        
        with get_session() as db:
            # Build the query with filters
            placeholders_sub = ", ".join([f":subcat_{i}" for i in range(len(request.selected_sub_categories))])
            placeholders_tf = ", ".join([f":tf_{i}" for i in range(len(request.selected_timeframes))])
            
            query = text(f"""
                SELECT DISTINCT tm.id, tm.name
                FROM target_metric tm
                JOIN target_metric_row tmr ON tm.id = tmr.target_metric_id
                JOIN target_metric_metadata_row mdr ON tmr.id = mdr.target_metric_row_id
                WHERE tm.target_metric_category_id = :category_id
                AND tm.sub_category IN ({placeholders_sub})
                AND mdr.name = 'Timeframe'
                AND mdr.value IN ({placeholders_tf})
            """)
            
            params = {"category_id": request.category_id}
            params.update({f"subcat_{i}": sub for i, sub in enumerate(request.selected_sub_categories)})
            params.update({f"tf_{i}": tf for i, tf in enumerate(request.selected_timeframes)})
            
            result = db.execute(query, params).fetchall()
            target_metric_ids = [row[0] for row in result]
        
        # Handle empty results
        if not target_metric_ids or not request.selected_metrics:
            logger.info("No target metric IDs or selected metrics, returning empty data")
            return {"setup_data": {}}
        
        # Get batch data
        batch_data = get_batch_setup_list_data(
            target_metric_ids, request.model_id, request.selected_metrics
        )
        
        # Convert any problematic types to safe types
        safe_batch_data = {}
        for key, value in batch_data.items():
            # Handle tuple keys properly
            if isinstance(key, tuple) and len(key) == 3:
                # Convert tuple to pipe-separated string format expected by client
                safe_key = f"{key[0]}|{key[1]}|{key[2]}"
            else:
                safe_key = str(key)
            
            if isinstance(value, dict):
                safe_value = {}
                for sub_key, sub_value in value.items():
                    safe_sub_key = str(sub_key)
                    # Convert numeric values safely
                    try:
                        if hasattr(sub_value, 'item'):  # numpy scalar
                            safe_sub_value = float(sub_value.item())
                        elif isinstance(sub_value, (int, float)):
                            safe_sub_value = float(sub_value)
                        else:
                            safe_sub_value = float(str(sub_value))
                    except (ValueError, TypeError):
                        safe_sub_value = 0.0
                    
                    safe_value[safe_sub_key] = safe_sub_value
            else:
                safe_value = str(value)
            
            safe_batch_data[safe_key] = safe_value
                
        return {"setup_data": safe_batch_data}
    except Exception as e:
        logger.error(f"Error fetching setup list data: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/dropdown-data")
async def get_dropdown_data():
    """Get all dropdown data in one optimized query with caching"""
    
    # Check cache first (10 minutes TTL since dropdown data rarely changes)
    cached_data = cache.get("dropdown_data", ttl_seconds=600)
    if cached_data:
        logger.info("Returning cached dropdown data")
        return cached_data
    
    try:
        logger.info("Cache miss - fetching dropdown data from database")
        start_time = time.time()
        
        with get_session() as db:
            all_data_query = db.execute(text("""
            -- Get regime models
            SELECT 'regime_model' as type, id, model_name as name, NULL as sub_category, NULL as timeframe
            FROM regime_model
            
            UNION ALL
            
            -- Get target metric categories  
            SELECT 'category' as type, id, name, NULL as sub_category, NULL as timeframe
            FROM target_metric_category
            
            UNION ALL
            
            -- Get distinct engineered metric names
            SELECT 'engineered_metric' as type, NULL as id, name, NULL as sub_category, NULL as timeframe
            FROM (SELECT DISTINCT name FROM engineered_target_metric_row) em
            
            UNION ALL
            
            -- Get timeframes
            SELECT 'timeframe' as type, NULL as id, mdr.value as name, NULL as sub_category, NULL as timeframe
            FROM target_metric_metadata_row mdr
            JOIN target_metric_row mr ON mdr.target_metric_row_id = mr.id
            WHERE mdr.name = 'Timeframe'
            GROUP BY mdr.value
            
            UNION ALL
            
            -- Get sub-categories
            SELECT 'sub_category' as type, NULL as id, sub_category as name, NULL as sub_category, NULL as timeframe
            FROM target_metric 
            WHERE sub_category IS NOT NULL
            GROUP BY sub_category
            """)).fetchall()
            
            # Process results
            regime_models = []
            categories = []
            engineered_metrics = []
            timeframes = []
            sub_categories = []
            
            for row in all_data_query:
                data_type, id_val, name, _, _ = row
                if data_type == 'regime_model':
                    regime_models.append({"id": id_val, "name": name})
                elif data_type == 'category':
                    categories.append({"id": id_val, "name": name})
                elif data_type == 'engineered_metric':
                    engineered_metrics.append(name)
                elif data_type == 'timeframe':
                    timeframes.append(name)
                elif data_type == 'sub_category':
                    sub_categories.append(name)
            
            # Get target metrics separately for setup combinations
            target_metrics_query = db.execute(text("""
                SELECT DISTINCT tm.id, tm.name, tm.sub_category, mdr.value as timeframe
                FROM target_metric tm
                JOIN target_metric_row tmr ON tm.id = tmr.target_metric_id
                JOIN target_metric_metadata_row mdr ON mdr.target_metric_row_id = tmr.id
                WHERE mdr.name = 'Timeframe'
                ORDER BY tm.name, tm.sub_category, mdr.value
            """)).fetchall()
            
            target_metrics = [{
                "id": row[0],
                "name": row[1], 
                "sub_category": row[2],
                "timeframe": row[3]
            } for row in target_metrics_query]
        
        # Build response
        response_data = {
            "regime_models": regime_models,
            "target_metric_categories": categories,
            "engineered_metrics": engineered_metrics,
            "timeframes": timeframes,
            "sub_categories": sub_categories,
            "target_metrics": target_metrics
        }
        
        # Cache the response
        cache.set("dropdown_data", response_data)
        
        end_time = time.time()
        logger.info(f"Dropdown data fetched and cached in {end_time - start_time:.2f} seconds")
        
        return response_data
    except Exception as e:
        logger.error(f"Error fetching dropdown data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/clear")
async def clear_cache():
    """Clear all cached data"""
    cache.clear()
    logger.info("Cache cleared manually")
    return {"message": "Cache cleared successfully"}

@app.get("/cache/status")
async def cache_status():
    """Get cache status information"""
    dropdown_cached = cache.get("dropdown_data", ttl_seconds=600) is not None
    return {
        "dropdown_data_cached": dropdown_cached,
        "cache_size": len(cache._cache)
    }

# =============================================================================
# CORRELATION ANALYSIS ENDPOINTS
# =============================================================================

@app.post("/correlation-analysis", response_model=List[CorrelationResult])
async def analyze_correlations(request: CorrelationRequest):
    """Perform correlation analysis for selected setup"""
    try:
        # Get current regime
        current_regime = get_current_regime(request.model_id)
        regime, regime_counts = get_regime_data(request.model_id)
        
        current_regime_name = current_regime.loc[current_regime['probability'].idxmax(), 'regime_name']
        current_regime_historical = regime[regime["regime_name"] == current_regime_name]
        timestamp_list = pd.to_datetime(current_regime_historical["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        
        with get_session() as db:
            # Get ALL target metric combinations
            all_combinations_query = db.execute(text("""
                SELECT DISTINCT 
                    tm.id,
                    tm.name,
                    tm.sub_category,
                    mdr.value as timeframe
                FROM target_metric tm
                JOIN target_metric_row tmr ON tm.id = tmr.target_metric_id
                JOIN target_metric_metadata_row mdr ON mdr.target_metric_row_id = tmr.id
                WHERE mdr.name = 'Timeframe'
                ORDER BY tm.name, tm.sub_category, mdr.value
            """)).fetchall()
        
            # Get data for selected setup
            selected_setup_id = db.execute(text("""
                SELECT DISTINCT tm.id
                FROM target_metric tm
                JOIN target_metric_row tmr ON tm.id = tmr.target_metric_id
                JOIN target_metric_metadata_row mdr ON mdr.target_metric_row_id = tmr.id
                WHERE tm.name = :name AND tm.sub_category = :sub_category 
                AND mdr.name = 'Timeframe' AND mdr.value = :timeframe
                LIMIT 1
            """), {
                "name": request.selected_setup_name,
                "sub_category": request.selected_sub_category, 
                "timeframe": request.selected_timeframe
            }).fetchone()[0]
        
            selected_data_query = db.execute(text("""
                SELECT timestamp, value
                FROM engineered_target_metric_row
                WHERE target_metric_id = :target_metric_id
                AND name = :metric_name
                AND timestamp IN :timestamps
                ORDER BY timestamp
            """), {
                "target_metric_id": selected_setup_id,
                "metric_name": request.engineered_metric_name,
                "timestamps": tuple(timestamp_list) if timestamp_list else tuple(['1900-01-01'])
            }).fetchall()
        
            if not selected_data_query:
                raise HTTPException(status_code=404, detail="No data found for selected setup in current regime")
            
            import pandas as pd
            selected_df = pd.DataFrame(selected_data_query, columns=['timestamp', 'value'])
            selected_df['timestamp'] = pd.to_datetime(selected_df['timestamp'])
            selected_series = selected_df.set_index('timestamp')['value']
            
            # Calculate correlations with all other setups
            correlation_results = []
            
            for target_metric_id, name, sub_category, timeframe in all_combinations_query:
                # Skip the selected setup itself
                if target_metric_id == selected_setup_id:
                    continue
                    
                other_data_query = db.execute(text("""
                    SELECT timestamp, value
                    FROM engineered_target_metric_row
                    WHERE target_metric_id = :target_metric_id
                    AND name = :metric_name
                    AND timestamp IN :timestamps
                    ORDER BY timestamp
                """), {
                    "target_metric_id": target_metric_id,
                    "metric_name": request.engineered_metric_name,
                    "timestamps": tuple(timestamp_list) if timestamp_list else tuple(['1900-01-01'])
                }).fetchall()
            
                if other_data_query:
                    other_df = pd.DataFrame(other_data_query, columns=['timestamp', 'value'])
                    other_df['timestamp'] = pd.to_datetime(other_df['timestamp'])
                    other_series = other_df.set_index('timestamp')['value']
                    
                    # Calculate correlation
                    try:
                        correlation = selected_series.corr(other_series)
                        if not pd.isna(correlation):
                            correlation_results.append(CorrelationResult(
                                setup=name,
                                sub_category=sub_category,
                                timeframe=timeframe,
                                setup_display=f"{name} | {sub_category} | {timeframe}",
                                correlation=correlation
                            ))
                    except:
                        continue
            
            return correlation_results
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# UTILITIES ENDPOINTS
# =============================================================================

class RoutineScriptRequest(BaseModel):
    job_type: str
    specific_models: List[str]

@app.post("/run-routine-script")
async def run_routine_script_endpoint(request: RoutineScriptRequest):
    """Run the routine script"""
    try:
        # Run in background to avoid blocking
        job_type = request.job_type
        specific_models = request.specific_models
        await asyncio.create_task(asyncio.to_thread(routine_script, job_type, specific_models))
        return {"status": "success", "message": "Routine script executed successfully"}
    except Exception as e:
        logger.error(f"Error running routine script: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SetupMatrixRequest(BaseModel):
    model_id: int
    setup_name: str
    sub_category: str
    timeframe: str
    engineered_metric_name: str

@app.post("/setup-matrix")
async def get_setup_matrix(request: SetupMatrixRequest):
    """Generate setup matrix showing correlations between selected setup and all other setups in current regime"""
    try:
        
        # Get current regime
        current_regime = get_current_regime(request.model_id)
        regime_df, regime_counts = get_regime_data(request.model_id)
        
        current_regime_name = current_regime.loc[current_regime['probability'].idxmax(), 'regime_name']
        current_regime_historical = regime_df[regime_df["regime_name"] == current_regime_name]
        timestamp_list = pd.to_datetime(current_regime_historical["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        
        with get_session() as db:
            # Get the selected target metric ID
            selected_target_query = text("""
                SELECT DISTINCT tm.id
                FROM target_metric tm
                JOIN target_metric_row tmr ON tm.id = tmr.target_metric_id
                JOIN target_metric_metadata_row mdr ON mdr.target_metric_row_id = tmr.id
                WHERE tm.name = :setup_name 
                AND tm.sub_category = :sub_category
                AND mdr.name = 'Timeframe' 
                AND mdr.value = :timeframe
                LIMIT 1
            """)
            
            selected_target_result = db.execute(selected_target_query, {
                "setup_name": request.setup_name,
                "sub_category": request.sub_category,
                "timeframe": request.timeframe
            }).fetchone()
        
            if not selected_target_result:
                return {
                    "correlations": [],
                    "message": "Selected setup not found"
                }
            
            selected_target_id = selected_target_result[0]
            
            # Get data for selected setup in current regime
            if not timestamp_list:
                return {
                    "correlations": [],
                    "message": "No current regime data available"
                }
            
            # Get all data for selected setup and filter in Python (more reliable)
            selected_data_query = text("""
                SELECT timestamp, value
                FROM engineered_target_metric_row
                WHERE target_metric_id = :target_metric_id
                AND name = :metric_name
                ORDER BY timestamp
            """)
            
            all_selected_data = db.execute(selected_data_query, {
                "target_metric_id": selected_target_id,
                "metric_name": request.engineered_metric_name
            }).fetchall()
        
            # Convert regime timestamps to set for fast lookup
            regime_dates = set(pd.to_datetime(timestamp_list).date)
            
            # Filter data to current regime dates
            selected_data_result = [
                row for row in all_selected_data 
                if pd.to_datetime(row[0]).date() in regime_dates
            ]
            
            if not selected_data_result:
                return {
                    "correlations": [],
                    "message": "No data found for selected setup in current regime"
                }
            
            # Convert to pandas series
            selected_df = pd.DataFrame(selected_data_result, columns=['timestamp', 'value'])
            selected_df['timestamp'] = pd.to_datetime(selected_df['timestamp'])
            selected_series = selected_df.set_index('timestamp')['value']
            
            # Get ALL other target metric combinations
            all_targets_query = text("""
                SELECT DISTINCT 
                    tm.id,
                    tm.name,
                    tm.sub_category,
                    mdr.value as timeframe
                FROM target_metric tm
                JOIN target_metric_row tmr ON tm.id = tmr.target_metric_id
                JOIN target_metric_metadata_row mdr ON mdr.target_metric_row_id = tmr.id
                WHERE mdr.name = 'Timeframe'
                ORDER BY tm.name, tm.sub_category, mdr.value
            """)
            
            all_targets = db.execute(all_targets_query).fetchall()
        
            # Calculate correlations with all other setups
            correlation_results = []
            
            for target_id, name, sub_category, timeframe in all_targets:
                # Skip the selected setup itself
                if target_id == selected_target_id:
                    continue
                
                # Get all data for this target and filter to current regime
                all_other_data = db.execute(selected_data_query, {
                    "target_metric_id": target_id,
                    "metric_name": request.engineered_metric_name
                }).fetchall()
            
                # Filter to current regime dates
                other_data_result = [
                    row for row in all_other_data 
                    if pd.to_datetime(row[0]).date() in regime_dates
                ]
                
                if other_data_result:
                    other_df = pd.DataFrame(other_data_result, columns=['timestamp', 'value'])
                    other_df['timestamp'] = pd.to_datetime(other_df['timestamp'])
                    other_series = other_df.set_index('timestamp')['value']
                    
                    # Calculate correlation
                    try:
                        correlation = selected_series.corr(other_series)
                        if not pd.isna(correlation):
                            correlation_results.append({
                                "setup_name": str(name),
                                "sub_category": str(sub_category),
                                "timeframe": str(timeframe),
                                "correlation": float(correlation),
                                "setup_display": f"{str(name)} | {str(sub_category)} | {str(timeframe)}"
                            })
                    except Exception as e:
                        logger.debug(f"Correlation calculation failed for {name}: {e}")
                        continue
        
        # Sort by correlation descending
        correlation_results.sort(key=lambda x: abs(x['correlation']), reverse=True)
                
        # Convert any numpy types to native Python types
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'dtype'):  # other numpy types
                return obj.item() if hasattr(obj, 'item') else str(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        response_data = {
            "correlations": convert_types(correlation_results),
            "selected_setup": f"{str(request.setup_name)} | {str(request.sub_category)} | {str(request.timeframe)}",
            "current_regime": convert_types(current_regime_name),
            "engineered_metric": str(request.engineered_metric_name),
            "model_id": int(request.model_id)
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error generating setup matrix: {e}")
        raise HTTPException(status_code=500, detail=f"Setup matrix generation failed: {str(e)}")

@app.get("/feature-engineering-functions")
async def get_feature_engineering_functions():
    """Get available feature engineering functions with their input requirements"""
    try:
        functions_with_requirements = {}
        for func_name in export_functions.keys():
            functions_with_requirements[func_name] = function_requirements.get(func_name, {
                "required_columns": ["unknown"],
                "description": "Function requirements not documented",
                "min_features": 1
            })
        
        return {
            "functions": list(export_functions.keys()),
            "function_details": functions_with_requirements
        }
    except Exception as e:
        logger.error(f"Error fetching feature engineering functions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class RunEngineeringFunctionRequest(BaseModel):
    category_id: int
    function_name: str

@app.post("/run-engineering-function")
async def run_engineering_function(request: RunEngineeringFunctionRequest):
    """Run engineering function on all target metrics in a category"""
    try:
        # Get function metadata fields
        function_metafield_input_lookup = {
            "hit_rate": ["P&L"],
            "pnl_per_day": ["P&L"],
            "P&L_per_direction": ["P&L", "Direction"],
            "total_pnl_per_day_per_direction": ["P&L", "Direction"],
            "profitable_trades_drawdown_per_direction": ["P&L", "Drawdown", "Direction"],
            "drawdown_per_direction": ["P&L", "Drawdown", "Direction"],
            "hit_rate_per_direction": ["P&L", "Direction"],
            "pnl_split_wins_losses": ["P&L"],
            "average_duration_by_type": ["P&L", "Direction", "Duration in Minutes"],
            "pnl_per_trade_per_direction_per_hour": ["P&L", "Direction"]
        }
        
        # Validate function exists
        if request.function_name not in function_metafield_input_lookup:
            raise HTTPException(status_code=400, detail=f"Function '{request.function_name}' not supported")
        
        metadata_fields = function_metafield_input_lookup[request.function_name]
        
        with get_session() as db:
            # Get target metrics in the category
            result = db.execute(text("""
                SELECT id FROM target_metric 
                WHERE target_metric_category_id = :category_id
            """), {"category_id": request.category_id}).fetchall()
            
            target_metric_ids = [row[0] for row in result]
            
            if not target_metric_ids:
                return {
                    "status": "warning",
                    "message": f"No target metrics found in category {request.category_id}",
                    "processed_count": 0
                }
            
            # Run the engineering function on each target metric
            processed_count = 0
            failed_metrics = []
            
            for target_metric_id in target_metric_ids:
                try:
                    run_backtest_function(db, request.function_name, target_metric_id, metadata_fields)
                    processed_count += 1
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"Failed to process target metric {target_metric_id}: {error_msg}")
                    failed_metrics.append({"target_metric_id": target_metric_id, "error": error_msg})
                    continue
        
        # Determine response status
        if processed_count == 0:
            status = "error"
            message = f"Engineering function '{request.function_name}' failed to process any target metrics"
        elif failed_metrics:
            status = "partial_success"
            message = f"Engineering function '{request.function_name}' completed with {len(failed_metrics)} failures"
        else:
            status = "success"
            message = f"Engineering function '{request.function_name}' completed successfully"
        
        response = {
            "status": status,
            "message": message,
            "processed_count": processed_count,
            "total_target_metrics": len(target_metric_ids)
        }
        
        # Include failure details if there were any
        if failed_metrics:
            response["failed_metrics"] = failed_metrics
            
        return response
        
    except Exception as e:
        logger.error(f"Error running engineering function: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# LIVE REGIME DATA ENDPOINT
# =============================================================================

@app.get("/live-regime-data/{model_id}/{target_metric_id}")
async def get_live_regime_data(model_id: int, target_metric_id: int, 
                              engineered_metric: str):
    """Get live regime data for probability window"""
    try:
        from target_metrics import get_live_regime_data
        
        result_df = get_live_regime_data(target_metric_id, model_id, engineered_metric)
        
        return {
            "regime_data": result_df.to_dict('records'),
            "model_id": model_id,
            "target_metric_id": target_metric_id,
            "engineered_metric": engineered_metric
        }
    except Exception as e:
        logger.error(f"Error fetching live regime data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# FEATURE MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/feature-categories")
async def get_feature_categories():
    """Get all feature categories"""
    try:
        with get_session() as db:
            result = db.execute(text("SELECT id, name FROM feature_category ORDER BY name")).fetchall()
            return [{"id": row[0], "name": row[1]} for row in result]
    except Exception as e:
        logger.error(f"Error fetching feature categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features")
async def get_features():
    """Get all features"""
    try:
        with get_session() as db:
            result = db.execute(text("""
                SELECT f.id, f.name, fc.name as category_name
                FROM feature f
                JOIN feature_category fc ON f.category_id = fc.id
                ORDER BY fc.name, f.name
            """)).fetchall()
            return [{"id": row[0], "name": row[1], "category": row[2]} for row in result]
    except Exception as e:
        logger.error(f"Error fetching features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class NewFeatureRequest(BaseModel):
    filename: str
    data_structure: str
    category_id: int
    sub_category: Optional[str] = None
    source: Optional[str] = None
    frequency: Optional[str] = None
    unit: Optional[str] = None
    plain_name: Optional[str] = None
    source_symbol: Optional[str] = None
    description: Optional[str] = None

@app.post("/features")
async def create_feature(
    file: UploadFile = File(...),
    data_structure: str = Form(...),
    category_id: int = Form(...),
    sub_category: str = Form(None),
    source: str = Form(None),
    frequency: str = Form(None),
    unit: str = Form(None),
    plain_name: str = Form(None),
    source_symbol: str = Form(None),
    description: str = Form(None)
):
    """Create a new feature"""
    import tempfile
    import os
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename or '')[1]) as temp_file:
            # Write uploaded file content to temporary file
            content = await file.read()
            temp_file.write(content)
            temp_filename = temp_file.name
        
        try:
            with get_session() as db:
                # Lookup category name from feature_category table
                category_result = db.execute(
                    text("SELECT name FROM feature_category WHERE id = :category_id"),
                    {"category_id": category_id}
                ).fetchone()
                category_name = category_result[0] if category_result else None

                # Read file into DataFrame
                import pandas as pd

                file_ext = os.path.splitext(file.filename or '')[1].lower()
                if file_ext == ".csv":
                    df = pd.read_csv(temp_filename)
                elif file_ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(temp_filename)
                else:
                    raise HTTPException(status_code=400, detail="Unsupported file type. Only CSV and Excel files are supported.")

                # Pass DataFrame to correct function
                if category_name == "macro_data":
                    macro_data(
                        df, data_structure, category_id, sub_category,
                        source, frequency, unit, plain_name,
                        source_symbol, description, db
                    )
                elif category_name == "market_data":
                    market_data(
                        df, data_structure, category_id, sub_category,
                        source, frequency, unit, plain_name,
                        source_symbol, description, db
                    )
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown category name: {category_name}")

            return {"status": "success", "message": "Feature created successfully"}
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_filename)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error creating feature: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/engineered-features")
async def get_engineered_features(include_dates: bool = False):
    """Get all engineered features, optionally with their earliest available dates"""
    try:
        with get_session() as db:
            result = db.execute(text("""
                SELECT ef.id, ef.name, ef.feature_engineering_id
                FROM engineered_feature ef
                ORDER BY ef.name
            """)).fetchall()
            
            features = []
            for row in result:
                feature_id, feature_name, feature_engineering_id = row
                
                feature_data = {
                    "id": feature_id,
                    "name": feature_name
                }
                
                # Only calculate earliest date if explicitly requested
                if include_dates:
                    earliest_date_result = db.execute(text("""
                        SELECT MAX(min_timestamp) as earliest_available_date
                        FROM (
                            SELECT MIN(fr.timestamp) as min_timestamp
                            FROM feature_engineering_row fer
                            JOIN feature_row fr ON fer.feature_id = fr.feature_id
                            WHERE fer.feature_engineering_id = :feature_engineering_id
                            GROUP BY fer.feature_id
                        ) AS min_dates
                    """), {"feature_engineering_id": feature_engineering_id}).fetchone()
                    
                    earliest_date = earliest_date_result[0] if earliest_date_result and earliest_date_result[0] else None
                    feature_data["earliest_available_date"] = earliest_date.isoformat() if earliest_date else None
                
                features.append(feature_data)
            
            return features
    except Exception as e:
        logger.error(f"Error fetching engineered features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class NewEngineeredFeatureRequest(BaseModel):
    name: str
    base_feature_ids: List[int]
    engineering_function: str

@app.post("/engineered-features")
async def create_engineered_feature(request: NewEngineeredFeatureRequest):
    """Create a new engineered feature with function and base features"""
    try:
        with get_session() as db:
            # 1. Insert into feature_engineering
            fe_query = text("""
                INSERT INTO feature_engineering (function)
                VALUES (:function)
                RETURNING id
            """)
            fe_result = db.execute(fe_query, {"function": request.engineering_function})
            feature_engineering_id = fe_result.fetchone()[0]

            # 2. Insert into feature_engineering_row for each base_feature_id
            for feature_id in request.base_feature_ids:
                fer_query = text("""
                    INSERT INTO feature_engineering_row (feature_id, feature_engineering_id)
                    VALUES (:feature_id, :feature_engineering_id)
                """)
                db.execute(fer_query, {
                    "feature_id": feature_id,
                    "feature_engineering_id": feature_engineering_id
                })

            # 3. Insert into engineered_feature
            ef_query = text("""
                INSERT INTO engineered_feature (name, feature_engineering_id)
                VALUES (:name, :feature_engineering_id)
                RETURNING id
            """)
            ef_result = db.execute(ef_query, {
                "name": request.name,
                "feature_engineering_id": feature_engineering_id
            })
            engineered_feature_id = ef_result.fetchone()[0]

        return {
            "status": "success",
            "engineered_feature_id": engineered_feature_id,
            "feature_engineering_id": feature_engineering_id,
            "message": "Engineered feature created successfully"
        }
    except Exception as e:
        logger.error(f"Error creating engineered feature: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# FEATURE ROW MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/features/{feature_id}/latest-timestamp")
async def get_latest_feature_timestamp(feature_id: int):
    """Get the latest timestamp for a specific feature"""
    try:
        with get_session() as db:
            result = db.execute(text("""
                SELECT MAX(timestamp) as latest_timestamp
                FROM feature_row 
                WHERE feature_id = :feature_id
            """), {"feature_id": feature_id}).fetchone()
            
            latest_timestamp = result[0] if result and result[0] else None
        
        return {
            "feature_id": feature_id,
            "latest_timestamp": latest_timestamp.isoformat() if latest_timestamp else None
        }
    except Exception as e:
        logger.error(f"Error fetching latest timestamp for feature {feature_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/{feature_id}/earliest-timestamp")
async def get_earliest_feature_timestamp(feature_id: int):
    """Get the earliest timestamp for a specific feature"""
    try:
        with get_session() as db:
            result = db.execute(text("""
                SELECT MIN(timestamp) as earliest_timestamp
                FROM feature_row 
                WHERE feature_id = :feature_id
            """), {"feature_id": feature_id}).fetchone()
            
            earliest_timestamp = result[0] if result and result[0] else None
        
        return {
            "feature_id": feature_id,
            "earliest_timestamp": earliest_timestamp.isoformat() if earliest_timestamp else None
        }
    except Exception as e:
        logger.error(f"Error fetching earliest timestamp for feature {feature_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class FeatureRowRequest(BaseModel):
    feature_id: int
    timestamp: str
    value: float

@app.post("/feature-rows")
async def add_feature_row(request: FeatureRowRequest):
    """Add or update a feature row"""
    try:
        from datetime import datetime
        
        # Parse timestamp
        try:
            timestamp = datetime.fromisoformat(request.timestamp.replace('Z', '+00:00'))
        except ValueError:
            # Try parsing as date string
            timestamp = datetime.strptime(request.timestamp, '%Y-%m-%d')
        
        with get_session() as db:
            # Insert or update feature row
            query = text("""
                INSERT INTO feature_row (feature_id, timestamp, value)
                VALUES (:feature_id, :timestamp, :value)
                ON CONFLICT (feature_id, timestamp) 
                DO UPDATE SET value = EXCLUDED.value
                RETURNING id
            """)
            
            result = db.execute(query, {
                "feature_id": request.feature_id,
                "timestamp": timestamp,
                "value": request.value
            })
            
            row_id = result.fetchone()[0]
        
        return {
            "status": "success",
            "message": "Feature row added/updated successfully",
            "row_id": row_id,
            "feature_id": request.feature_id,
            "timestamp": timestamp.isoformat(),
            "value": request.value
        }
        
    except Exception as e:
        logger.error(f"Error adding feature row: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/{feature_id}/rows")
async def get_feature_rows(feature_id: int, limit: int = 100):
    """Get recent feature rows for a specific feature"""
    try:
        with get_session() as db:
            result = db.execute(text("""
                SELECT id, timestamp, value
                FROM feature_row 
                WHERE feature_id = :feature_id
                ORDER BY timestamp DESC
                LIMIT :limit
            """), {"feature_id": feature_id, "limit": limit}).fetchall()
            
            rows = []
            for row in result:
                rows.append({
                    "id": row[0],
                    "timestamp": row[1].isoformat() if row[1] else None,
                    "value": float(row[2]) if row[2] is not None else None
                })
        
        return {
            "feature_id": feature_id,
            "rows": rows,
            "count": len(rows)
        }
    except Exception as e:
        logger.error(f"Error fetching feature rows for feature {feature_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# DATASET-FEATURE RELATIONSHIP ENDPOINTS
# =============================================================================

@app.get("/dataset-features/{dataset_id}")
async def get_dataset_features(dataset_id: int):
    """Get features associated with a dataset"""
    try:
        with get_session() as db:
            result = db.execute(text("""
                SELECT ef.id, ef.name
                FROM dataset_row dr
                JOIN engineered_feature ef ON dr.engineered_feature_id = ef.id
                WHERE dr.dataset_id = :dataset_id
                ORDER BY ef.name
            """), {"dataset_id": dataset_id}).fetchall()
            
            return [{"id": row[0], "name": row[1]} for row in result]
    except Exception as e:
        logger.error(f"Error fetching dataset features: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel
from typing import List

class AddFeaturesToDatasetRequest(BaseModel):
    regime_model_id: int
    engineered_feature_ids: List[int]

@app.post("/dataset-features")
async def add_features_to_dataset(request: AddFeaturesToDatasetRequest):
    """Add multiple engineered features to a dataset based on regime model"""
    try:
        with get_session() as db:
            # 1. Get dataset_id from regime_model
            result = db.execute(
                text("SELECT dataset_id FROM regime_model WHERE id = :regime_model_id"),
                {"regime_model_id": request.regime_model_id}
            ).fetchone()
            if not result or not result[0]:
                raise HTTPException(status_code=404, detail="Dataset not found for selected regime model")
            dataset_id = result[0]

            # 2. Insert each engineered_feature_id into dataset_engineered_feature_relationship
            inserted_ids = []
            for feature_id in request.engineered_feature_ids:
                # Check if relationship already exists
                check = db.execute(
                    text("""
                        SELECT id FROM dataset_row
                        WHERE dataset_id = :dataset_id AND engineered_feature_id = :feature_id
                    """),
                    {"dataset_id": dataset_id, "feature_id": feature_id}
                ).fetchone()
                if check:
                    continue  # Skip if already exists

                insert = db.execute(
                    text("""
                        INSERT INTO dataset_row (dataset_id, engineered_feature_id)
                        VALUES (:dataset_id, :feature_id)
                        RETURNING id
                    """),
                    {"dataset_id": dataset_id, "feature_id": feature_id}
                )
                inserted_id = insert.fetchone()[0]
                inserted_ids.append(inserted_id)

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "inserted_relationship_ids": inserted_ids,
            "message": f"{len(inserted_ids)} features added to dataset"
        }
    except Exception as e:
        logger.error(f"Error adding features to dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# DATASET-REGIME RELATIONSHIP METHODS
# =============================================================================

class AddRegimeToDatasetRequest(BaseModel):
    regime_model_id: int
    regime_model_ids: List[int]

@app.post("/dataset-regimes")
async def add_regimes_to_dataset(request: AddRegimeToDatasetRequest):
    """Add regime models as features to a dataset"""
    try:
        with get_session() as db:
            # 1. Get dataset_id from regime_model
            result = db.execute(
                text("SELECT dataset_id FROM regime_model WHERE id = :regime_model_id"),
                {"regime_model_id": request.regime_model_id}
            ).fetchone()
            if not result or not result[0]:
                raise HTTPException(status_code=404, detail="Dataset not found for selected regime model")
            dataset_id = result[0]

            # 2. Insert each regime_model_id into dataset_row_regime
            inserted_ids = []
            skipped_self_references = []
            
            for regime_model_id in request.regime_model_ids:
                # Prevent self-referential regime features (regime model adding itself as feature)
                if regime_model_id == request.regime_model_id:
                    skipped_self_references.append(regime_model_id)
                    continue
                
                # Check if relationship already exists
                check = db.execute(
                    text("""
                        SELECT id FROM dataset_row_regime
                        WHERE dataset_id = :dataset_id AND regime_model_id = :regime_model_id
                    """),
                    {"dataset_id": dataset_id, "regime_model_id": regime_model_id}
                ).fetchone()
                if check:
                    continue  # Skip if already exists

                insert = db.execute(
                    text("""
                        INSERT INTO dataset_row_regime (dataset_id, regime_model_id)
                        VALUES (:dataset_id, :regime_model_id)
                        RETURNING id
                    """),
                    {"dataset_id": dataset_id, "regime_model_id": regime_model_id}
                )
                inserted_id = insert.fetchone()[0]
                inserted_ids.append(inserted_id)
        
        # Build response message
        message_parts = []
        if inserted_ids:
            message_parts.append(f"{len(inserted_ids)} regime models added to dataset as features")
        
        if skipped_self_references:
            message_parts.append(f"Skipped {len(skipped_self_references)} self-referential regime model(s) - cannot add a regime model as feature to its own dataset")
        
        if not inserted_ids and not skipped_self_references:
            message_parts.append("No new regime models were added (all relationships already exist)")
            
        return {
            "status": "success",
            "dataset_id": dataset_id,
            "inserted_relationship_ids": inserted_ids,
            "skipped_self_references": skipped_self_references,
            "message": ". ".join(message_parts)
        }
    except Exception as e:
        logger.error(f"Error adding regime models to dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset-regimes/{dataset_id}")
async def get_dataset_regimes(dataset_id: int):
    """Get regime models associated with a dataset"""
    try:
        with get_session() as db:
            result = db.execute(text("""
                SELECT drr.id, drr.regime_model_id, rm.model_name as regime_model_name
                FROM dataset_row_regime drr
                JOIN regime_model rm ON drr.regime_model_id = rm.id
                WHERE drr.dataset_id = :dataset_id
            """), {"dataset_id": dataset_id}).fetchall()
            
            regimes = []
            for row in result:
                regimes.append({
                    "id": row[0],
                    "regime_model_id": row[1], 
                    "regime_model_name": row[2]
                })
        
        return regimes
    except Exception as e:
        logger.error(f"Error fetching dataset regimes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# STATISTICAL ANALYSIS ENDPOINT
# =============================================================================

class StatisticalAnalysisRequest(BaseModel):
    target_metric_id: int
    model_id: int
    engineered_metric_name: str
    timeframe: Optional[str] = None

@app.post("/statistical-analysis")
async def perform_statistical_analysis(request: StatisticalAnalysisRequest):
    """Comprehensive statistical analysis for confirmation window using engineered target metrics"""
    try:
        from target_metrics import get_target_metric_data, get_current_regime, get_regime_data
        import pandas as pd
        import numpy as np
        from scipy import stats
        from statsmodels.tsa.stattools import adfuller, kpss
        from statsmodels.stats.diagnostic import acorr_ljungbox
                
        # Get engineered target metric data
        target_data = get_target_metric_data(request.target_metric_id, request.engineered_metric_name)
        
        # Get current regime and historical regime data
        current_regime = get_current_regime(request.model_id)
        regime_data, regime_counts = get_regime_data(request.model_id)
        
        current_regime_name = current_regime.loc[current_regime['probability'].idxmax(), 'regime_name']
        
        # Filter data for current regime periods with forward-looking alignment
        current_regime_historical = regime_data[regime_data["regime_name"] == current_regime_name]
        # Apply shift(1) for forward-looking regime alignment (regime on day T predicts performance on day T+1)
        regime_timestamps = pd.to_datetime(current_regime_historical["timestamp"]).shift(1).dropna().dt.date
        target_data['date'] = pd.to_datetime(target_data['timestamp']).dt.date
                
        # Filter target data to only include current regime periods
        filtered_data = target_data[target_data['date'].isin(regime_timestamps)].copy()
        
        if filtered_data.empty:
            return {"status": "error", "message": "No data available for current regime analysis"}
        
        # Prepare values for analysis
        values = filtered_data['value'].dropna()
        
        if len(values) < 10:
            return {"status": "error", "message": "Insufficient data points for statistical analysis"}
        
        # Basic statistics with enhanced calculations
        mean_val = float(values.mean())
        std_val = float(values.std())
        min_val = float(values.min())
        max_val = float(values.max())
        
        # Z-scores for min/max
        min_z = (min_val - mean_val) / std_val if std_val != 0 else 0
        max_z = (max_val - mean_val) / std_val if std_val != 0 else 0
        
        # Probability of observing such extreme values
        prob_min = stats.norm.sf(abs(min_z)) * 2  # Two-tailed
        prob_max = stats.norm.sf(abs(max_z)) * 2
        
        basic_stats = {
            "count": int(len(values)),
            "mean": mean_val,
            "std": std_val,
            "skewness": float(values.skew()),
            "kurtosis": float(values.kurtosis()),
            "min": min_val,
            "max": max_val,
            "min_zscore": float(min_z),
            "max_zscore": float(max_z),
            "min_probability": float(prob_min),
            "max_probability": float(prob_max)
        }
        
        # Comprehensive percentiles
        percentiles = [0.1, 0.5, 1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99, 99.5, 99.9]
        percentile_data = {}
        for p in percentiles:
            percentile_data[f"p{p}"] = float(values.quantile(p/100))
        
        # Stationarity tests
        adf_stat, adf_pvalue, _, _, adf_critical, _ = adfuller(values)
        kpss_stat, kpss_pvalue, _, kpss_critical = kpss(values, regression='c', nlags="auto")
        
        # Normality test (limit to 5000 samples for performance)
        test_sample = values[:5000] if len(values) > 5000 else values
        shapiro_stat, shapiro_pvalue = stats.shapiro(test_sample)
        
        # Ljung-Box test for autocorrelation
        ljung_box_result = acorr_ljungbox(values, lags=min(10, len(values)//4), return_df=True)
        
        # Change point detection (basic implementation)
        try:
            import ruptures as rpt
            change_points = rpt.Pelt(model="rbf").fit(values.values).predict(pen=10)
            change_points = [int(cp) for cp in change_points[:-1]]  # Remove last point (end of series)
        except:
            change_points = []
        
        # Rolling statistics for plotting
        window_size = min(20, len(values)//5)
        if window_size > 1:
            filtered_data = filtered_data.copy()
            filtered_data['rolling_mean'] = filtered_data['value'].rolling(window_size).mean()
            filtered_data['rolling_std'] = filtered_data['value'].rolling(window_size).std()
        
        # Time series data for plotting
        time_series_data = filtered_data[['timestamp', 'value']].copy()
        if 'rolling_mean' in filtered_data.columns:
            time_series_data['rolling_mean'] = filtered_data['rolling_mean']
            time_series_data['rolling_std'] = filtered_data['rolling_std']
            time_series_data['upper_band'] = filtered_data['rolling_mean'] + filtered_data['rolling_std']
            time_series_data['lower_band'] = filtered_data['rolling_mean'] - filtered_data['rolling_std']
        
        # Convert to JSON-safe format
        time_series_records = []
        for _, row in time_series_data.iterrows():
            record = {
                'timestamp': row['timestamp'].isoformat() if pd.notna(row['timestamp']) else None,
                'value': float(row['value']) if pd.notna(row['value']) else None
            }
            if 'rolling_mean' in row and pd.notna(row['rolling_mean']):
                record['rolling_mean'] = float(row['rolling_mean'])
                record['rolling_std'] = float(row['rolling_std'])
                record['upper_band'] = float(row['upper_band'])
                record['lower_band'] = float(row['lower_band'])
            time_series_records.append(record)
        
        # Prepare the response data
        response_data = {
            "status": "success",
            "basic_statistics": basic_stats,
            "percentiles": percentile_data,
            "stationarity_tests": {
                "adf_statistic": float(adf_stat),
                "adf_pvalue": float(adf_pvalue),
                "adf_critical_values": {str(k): float(v) for k, v in adf_critical.items()},
                "kpss_statistic": float(kpss_stat),
                "kpss_pvalue": float(kpss_pvalue),
                "kpss_critical_values": {str(k): float(v) for k, v in kpss_critical.items()}
            },
            "normality_test": {
                "shapiro_statistic": float(shapiro_stat),
                "shapiro_pvalue": float(shapiro_pvalue)
            },
            "autocorrelation_test": ljung_box_result.to_dict('records'),
            "change_points": change_points,
            "time_series_data": time_series_records,
            "current_regime": str(current_regime_name),  # Ensure string conversion
            "data_points_analyzed": len(values),
            "regime_filter_applied": True
        }
        
        # Convert all numpy types to JSON-safe types
        return convert_numpy_types(response_data)
        
    except Exception as e:
        logger.error(f"Error performing statistical analysis: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Statistical analysis error: {str(e)}")

# =============================================================================
# TRADE PLAN ANALYSIS ENDPOINTS
# =============================================================================

class SetupData(BaseModel):
    name: str
    sub_category: str
    timeframe: str
    direction: str

class TradePlanRequest(BaseModel):
    regime_model_id: int
    contract_multiplier: float
    point_value: float
    volatility_multiplier: float
    selected_setups: List[SetupData]

@app.post("/analyze-trade-plan")
async def analyze_trade_plan(request: TradePlanRequest):
    """Analyze trade plan with P&L statistics for current regime"""
    try:
        
        # Get current regime
        current_regime_df = get_current_regime(request.regime_model_id)
        if current_regime_df.empty:
            raise HTTPException(status_code=404, detail="No current regime data found")
        
        # Get the current regime name with highest probability
        current_regime_name = current_regime_df.loc[current_regime_df['probability'].idxmax(), 'regime_name']
        
        # Get regime data for filtering (this already contains only the highest probability regime per timestamp)
        regime_df, _ = get_regime_data(request.regime_model_id)
        regime_df["timestamp"] = pd.to_datetime(regime_df["timestamp"])
        
        # Filter for dates when the current regime had the highest probability
        # Shift by 1 to align regime with forward-looking trade performance
        current_regime_timestamps = regime_df[regime_df["regime_name"] == current_regime_name]["timestamp"]
        current_regime_dates = set(current_regime_timestamps.shift(1).dropna().dt.date)
                
        setup_analysis = []
        
        # Analyze each selected setup
        for setup in request.selected_setups:
            try:                
                # Find target metric ID - try multiple timeframe formats
                target_metric_query = None
                
                # Try different timeframe formats that might be in the database
                timeframe_variations = [
                    setup.timeframe,  # Original format (e.g., "5m")
                    f"{setup.timeframe.replace('m', ' minutes').replace('h', ' hour')}",  # "5 minutes", "1 hour"
                    f"{setup.timeframe.replace('m', ' minute').replace('h', ' hour') if setup.timeframe.endswith('m') and setup.timeframe[:-1] == '1' else setup.timeframe.replace('m', ' minutes').replace('h', ' hour')}"  # Handle singular vs plural
                ]
                
                for tf_variant in timeframe_variations:
                    target_metric_query = db.execute(text("""
                        SELECT DISTINCT tm.id
                        FROM target_metric tm
                        JOIN target_metric_row tmr ON tm.id = tmr.target_metric_id
                        JOIN target_metric_metadata_row mdr ON mdr.target_metric_row_id = tmr.id
                        WHERE tm.name = :name 
                        AND tm.sub_category = :sub_category
                        AND mdr.name = 'Timeframe' 
                        AND mdr.value = :timeframe
                        LIMIT 1
                    """), {
                        "name": setup.name,
                        "sub_category": setup.sub_category,
                        "timeframe": tf_variant
                    }).fetchone()
                    
                    if target_metric_query:
                        break
                
                if not target_metric_query:
                    setup_analysis.append({
                        "setup": setup.model_dump(),
                        "statistics": None,
                        "error": f"Target metric not found for {setup.name} | {setup.sub_category} | {setup.timeframe}"
                    })
                    continue
                
                target_metric_id = target_metric_query[0]
                
                # Get P&L data for the specific direction in current regime
                pnl_query = db.execute(text("""
                    SELECT tmr.timestamp, mdr_pnl.value as pnl
                    FROM target_metric_row tmr
                    JOIN target_metric_metadata_row mdr_pnl ON mdr_pnl.target_metric_row_id = tmr.id
                    JOIN target_metric_metadata_row mdr_dir ON mdr_dir.target_metric_row_id = tmr.id
                    WHERE tmr.target_metric_id = :target_metric_id
                    AND mdr_pnl.name = 'P&L'
                    AND mdr_dir.name = 'Direction'
                    AND mdr_dir.value = :direction
                """), {
                    "target_metric_id": target_metric_id,
                    "direction": setup.direction
                }).fetchall()
                
                if not pnl_query:
                    setup_analysis.append({
                        "setup": setup.model_dump(),
                        "statistics": None,
                        "error": f"No P&L data found for {setup.direction} direction"
                    })
                    continue
                
                # Convert to DataFrame and filter for current regime
                pnl_df = pd.DataFrame(pnl_query, columns=['timestamp', 'pnl'])
                pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])
                pnl_df['date'] = pnl_df['timestamp'].dt.date
                pnl_df['pnl'] = pd.to_numeric(pnl_df['pnl'], errors='coerce')
                
                # Filter for current regime dates
                current_regime_pnl = pnl_df[pnl_df['date'].isin(current_regime_dates)]
                
                if current_regime_pnl.empty:
                    setup_analysis.append({
                        "setup": setup.model_dump(),
                        "statistics": None,
                        "error": "No data available for current regime"
                    })
                    continue
                
                # Calculate statistics
                pnl_values = current_regime_pnl['pnl'].dropna()
                
                if len(pnl_values) == 0:
                    setup_analysis.append({
                        "setup": setup.model_dump(),
                        "statistics": None,
                        "error": "No valid P&L values for current regime"
                    })
                    continue
                
                # Calculate mean, std, percentiles, and skewness
                pnl_stats = {
                    "mean": float(pnl_values.mean()),
                    "std": float(pnl_values.std()),
                    "skewness": float(stats.skew(pnl_values)),
                    "p1": float(np.percentile(pnl_values, 1)),
                    "p2_5": float(np.percentile(pnl_values, 2.5)),
                    "p5": float(np.percentile(pnl_values, 5)),
                    "p10": float(np.percentile(pnl_values, 10)),
                    "p25": float(np.percentile(pnl_values, 25)),
                    "count": int(len(pnl_values))
                }
                
                setup_analysis.append({
                    "setup": setup.model_dump(),
                    "statistics": pnl_stats,
                    "error": None
                })
                                
            except Exception as e:
                logger.error(f"Error analyzing setup {setup.name}: {e}")
                setup_analysis.append({
                    "setup": setup.model_dump(),
                    "statistics": None,
                    "error": str(e)
                })
        
        result = {
            "regime_model_id": request.regime_model_id,
            "current_regime": current_regime_name,
            "setup_analysis": setup_analysis,
            "parameters": {
                "contract_multiplier": request.contract_multiplier,
                "point_value": request.point_value,
                "volatility_multiplier": request.volatility_multiplier
            }
        }
        
        return convert_numpy_types(result)
        
    except Exception as e:
        logger.error(f"Error in trade plan analysis: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Trade plan analysis error: {str(e)}")

@app.get("/comprehensive-regime-analysis/{model_id}")
async def get_comprehensive_regime_analysis(model_id: int):
    """Get comprehensive regime analysis including model info and dataset features"""
    try:
        from regime_model import build_dataset
        import json
        
        # Get model information first (quick query)
        with get_session() as db:
            model_query = text("SELECT * FROM regime_model WHERE id = :model_id")
            model_result = db.execute(model_query, {"model_id": model_id}).fetchone()
        
        dataset_records = []
        model_params = {}
        
        if model_result:
            dataset_id = model_result[1]
            model_type = model_result[2]
            
            try:
                # Parse model parameters if they exist
                if model_result[3]:
                    if isinstance(model_result[3], str):
                        model_params = json.loads(model_result[3])
                    else:
                        model_params = model_result[3]
            except:
                model_params = {}
        
        # Get regime data using the existing function (separate session)
        regime_df, regime_counts = get_historical_regime_data(model_id)
        
        # Get dataset features if model exists
        if model_result:
            try:
                dataset_data = build_dataset(dataset_id, model_type)
                # Convert dataset to serializable format
                dataset_records = convert_numpy_types(dataset_data.to_dict('records'))
            except Exception as e:
                logger.warning(f"Could not build dataset for model {model_id}: {e}")
                dataset_records = []
        
        return {
            "regime_data": regime_df.to_dict('records'),
            "regime_counts": regime_counts,
            "dataset_data": dataset_records,
            "model_params": model_params
        }
        
    except Exception as e:
        logger.error(f"Error fetching comprehensive regime analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# SERVER STARTUP
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )