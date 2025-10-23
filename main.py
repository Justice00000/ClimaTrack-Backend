# main.py - ClimaTrack Backend API
"""
ClimaTrack Backend Server
Predictive Waterborne Disease Risk Monitoring System
Author: Chukwuonye Justice Izuchukwu
"""

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import uvicorn
import jwt
import bcrypt
from geopy.distance import geodesic
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

SECRET_KEY = "k46cIo6jN4zRe6RkmWUGKYy5HQ7EtkVXxg2_5VaDvNM"  # Change this!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(
    title="ClimaTrack API",
    description="AI-Driven Waterborne Disease Risk Prediction Platform",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class ReportType(str, Enum):
    WATER_QUALITY = "water_quality"
    SANITATION = "sanitation"
    HEALTH_SYMPTOM = "health_symptom"
    INFRASTRUCTURE = "infrastructure"

class AlertType(str, Enum):
    RISK_INCREASE = "risk_increase"
    OUTBREAK_WARNING = "outbreak_warning"
    WEATHER_ALERT = "weather_alert"
    WATER_CONTAMINATION = "water_contamination"

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class UserRegistration(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(..., min_length=6)
    phone: Optional[str] = None
    location: Optional[Dict[str, float]] = None  # {"latitude": 6.5244, "longitude": 3.3792}

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_info: Dict[str, Any]

class LocationData(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    accuracy: Optional[float] = None

class EnvironmentalData(BaseModel):
    location: LocationData
    ndwi: Optional[float] = Field(None, description="Normalized Difference Water Index")
    awei: Optional[float] = Field(None, description="Automated Water Extraction Index")
    rainfall: Optional[float] = Field(None, description="Rainfall in mm")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    humidity: Optional[float] = Field(None, description="Humidity percentage")
    flood_risk: Optional[float] = Field(None, ge=0, le=1)

class RiskPrediction(BaseModel):
    location: LocationData
    risk_level: RiskLevel
    risk_score: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    contributing_factors: List[Dict[str, Any]]
    prediction_date: datetime
    valid_until: datetime
    recommendations: List[str]

class CommunityReport(BaseModel):
    report_type: ReportType
    location: LocationData
    description: str = Field(..., min_length=10, max_length=500)
    severity: Optional[int] = Field(None, ge=1, le=5)
    images: Optional[List[str]] = None  # URLs to uploaded images
    reporter_contact: Optional[str] = None

class HealthSymptom(BaseModel):
    symptom_type: str
    severity: int = Field(..., ge=1, le=5)
    onset_date: datetime
    location: LocationData
    affected_count: int = Field(default=1, ge=1)
    additional_info: Optional[str] = None

class AlertNotification(BaseModel):
    alert_type: AlertType
    risk_level: RiskLevel
    title: str
    message: str
    location: LocationData
    affected_radius: float  # in kilometers
    issued_at: datetime
    expires_at: Optional[datetime] = None
    action_items: List[str]

class ZoneData(BaseModel):
    zone_id: str
    name: str
    location: LocationData
    risk_level: RiskLevel
    population: Optional[int] = None
    last_updated: datetime
    historical_outbreaks: Optional[int] = None

# ============================================================================
# IN-MEMORY DATABASE (Replace with PostgreSQL/MongoDB in production)
# ============================================================================

USERS_DB = {}  # email -> user_data
PREDICTIONS_DB = {}  # location_key -> prediction_data
REPORTS_DB = []  # List of community reports
ALERTS_DB = []  # List of active alerts
HEALTH_LOGS_DB = []  # List of health symptoms

# Sample zones for Lagos area
ZONES_DB = {
    "ikeja_north": {
        "zone_id": "ikeja_north",
        "name": "Ikeja North",
        "location": {"latitude": 6.6149, "longitude": 3.3406},
        "risk_level": RiskLevel.LOW,
        "population": 85000,
        "last_updated": datetime.now(),
        "historical_outbreaks": 2
    },
    "oshodi": {
        "zone_id": "oshodi",
        "name": "Oshodi",
        "location": {"latitude": 6.5586, "longitude": 3.3469},
        "risk_level": RiskLevel.LOW,
        "population": 120000,
        "last_updated": datetime.now(),
        "historical_outbreaks": 3
    },
    "victoria_island": {
        "zone_id": "victoria_island",
        "name": "Victoria Island",
        "location": {"latitude": 6.4281, "longitude": 3.4219},
        "risk_level": RiskLevel.MODERATE,
        "population": 95000,
        "last_updated": datetime.now(),
        "historical_outbreaks": 1
    },
    "surulere": {
        "zone_id": "surulere",
        "name": "Surulere",
        "location": {"latitude": 6.4969, "longitude": 3.3614},
        "risk_level": RiskLevel.HIGH,
        "population": 150000,
        "last_updated": datetime.now(),
        "historical_outbreaks": 5
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return email
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def calculate_distance(loc1: Dict[str, float], loc2: Dict[str, float]) -> float:
    """Calculate distance between two coordinates in kilometers"""
    coords_1 = (loc1["latitude"], loc1["longitude"])
    coords_2 = (loc2["latitude"], loc2["longitude"])
    return geodesic(coords_1, coords_2).kilometers

def generate_location_key(location: LocationData) -> str:
    """Generate unique key for location (rounded to ~1km grid)"""
    lat_rounded = round(location.latitude, 2)
    lon_rounded = round(location.longitude, 2)
    return f"{lat_rounded},{lon_rounded}"

def calculate_risk_score(environmental_data: EnvironmentalData) -> tuple:
    """
    ML-based risk prediction (simplified version)
    In production, this would use trained TensorFlow/PyTorch model
    """
    score = 0.0
    factors = []
    
    # Water index analysis
    if environmental_data.ndwi and environmental_data.ndwi > 0.3:
        score += 0.2
        factors.append({"factor": "Standing water detected", "impact": 0.2})
    
    # Rainfall analysis
    if environmental_data.rainfall and environmental_data.rainfall > 50:
        score += 0.25
        factors.append({"factor": "Heavy rainfall", "impact": 0.25})
    
    # Temperature analysis (optimal for pathogen growth: 25-35°C)
    if environmental_data.temperature and 25 <= environmental_data.temperature <= 35:
        score += 0.15
        factors.append({"factor": "Optimal temperature for pathogens", "impact": 0.15})
    
    # Humidity analysis
    if environmental_data.humidity and environmental_data.humidity > 70:
        score += 0.1
        factors.append({"factor": "High humidity", "impact": 0.1})
    
    # Flood risk
    if environmental_data.flood_risk and environmental_data.flood_risk > 0.5:
        score += 0.3
        factors.append({"factor": "Flood risk detected", "impact": 0.3})
    
    # Normalize score
    score = min(score, 1.0)
    
    # Determine risk level
    if score < 0.3:
        risk_level = RiskLevel.LOW
    elif score < 0.6:
        risk_level = RiskLevel.MODERATE
    elif score < 0.8:
        risk_level = RiskLevel.HIGH
    else:
        risk_level = RiskLevel.CRITICAL
    
    return risk_level, score, factors

def generate_recommendations(risk_level: RiskLevel, factors: List[Dict]) -> List[str]:
    """Generate actionable recommendations based on risk level"""
    recommendations = []
    
    if risk_level == RiskLevel.LOW:
        recommendations.append("Maintain good hygiene practices")
        recommendations.append("Ensure water storage containers are clean")
        recommendations.append("Continue regular water quality monitoring")
    
    elif risk_level == RiskLevel.MODERATE:
        recommendations.append("Boil drinking water for at least 1 minute")
        recommendations.append("Wash hands frequently with soap")
        recommendations.append("Avoid contact with stagnant water")
        recommendations.append("Store water in covered containers")
    
    elif risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
        recommendations.append("⚠️ URGENT: Boil all water before drinking")
        recommendations.append("⚠️ Use water purification tablets if available")
        recommendations.append("⚠️ Avoid open water sources")
        recommendations.append("⚠️ Seek medical attention if symptoms develop")
        recommendations.append("⚠️ Report water contamination to authorities")
        recommendations.append("Practice strict hand hygiene")
    
    return recommendations

# ============================================================================
# API ENDPOINTS - AUTHENTICATION
# ============================================================================

@app.post("/api/auth/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegistration):
    """Register new user account"""
    
    # Check if user already exists
    if user_data.email in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash password
    hashed_password = hash_password(user_data.password)
    
    # Store user
    user_info = {
        "email": user_data.email,
        "full_name": user_data.full_name,
        "password": hashed_password,
        "phone": user_data.phone,
        "location": user_data.location,
        "created_at": datetime.now().isoformat(),
        "is_active": True
    }
    
    USERS_DB[user_data.email] = user_info
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_data.email}, expires_delta=access_token_expires
    )
    
    # Remove password from response
    user_response = {k: v for k, v in user_info.items() if k != "password"}
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_info": user_response
    }

@app.post("/api/auth/login", response_model=Token)
async def login_user(credentials: UserLogin):
    """Authenticate user and return access token"""
    
    # Check if user exists
    if credentials.email not in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    user = USERS_DB[credentials.email]
    
    # Verify password
    if not verify_password(credentials.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": credentials.email}, expires_delta=access_token_expires
    )
    
    # Remove password from response
    user_response = {k: v for k, v in user.items() if k != "password"}
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_info": user_response
    }

@app.get("/api/auth/me")
async def get_current_user(email: str = Depends(verify_token)):
    """Get current authenticated user information"""
    if email not in USERS_DB:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = USERS_DB[email]
    return {k: v for k, v in user.items() if k != "password"}

# ============================================================================
# API ENDPOINTS - RISK PREDICTION
# ============================================================================

@app.post("/api/predictions/generate", response_model=RiskPrediction)
async def generate_risk_prediction(
    env_data: EnvironmentalData,
    email: str = Depends(verify_token)
):
    """Generate waterborne disease risk prediction for location"""
    
    # Calculate risk using ML model (simplified version)
    risk_level, risk_score, factors = calculate_risk_score(env_data)
    
    # Generate recommendations
    recommendations = generate_recommendations(risk_level, factors)
    
    # Create prediction
    prediction = RiskPrediction(
        location=env_data.location,
        risk_level=risk_level,
        risk_score=risk_score,
        confidence=0.85,  # In production, from ML model
        contributing_factors=factors,
        prediction_date=datetime.now(),
        valid_until=datetime.now() + timedelta(hours=24),
        recommendations=recommendations
    )
    
    # Store prediction
    location_key = generate_location_key(env_data.location)
    PREDICTIONS_DB[location_key] = prediction.dict()
    
    return prediction

@app.get("/api/predictions/location", response_model=RiskPrediction)
async def get_prediction_by_location(
    latitude: float,
    longitude: float,
    email: str = Depends(verify_token)
):
    """Get risk prediction for specific location"""
    
    location = LocationData(latitude=latitude, longitude=longitude)
    location_key = generate_location_key(location)
    
    if location_key in PREDICTIONS_DB:
        prediction = PREDICTIONS_DB[location_key]
        
        # Check if prediction is still valid
        valid_until = datetime.fromisoformat(prediction["valid_until"])
        if datetime.now() < valid_until:
            return prediction
    
    # Generate new prediction if not found or expired
    env_data = EnvironmentalData(
        location=location,
        rainfall=np.random.uniform(0, 100),
        temperature=np.random.uniform(20, 35),
        humidity=np.random.uniform(50, 90),
        ndwi=np.random.uniform(0, 0.5),
        flood_risk=np.random.uniform(0, 1)
    )
    
    return await generate_risk_prediction(env_data, email)

@app.get("/api/predictions/nearby")
async def get_nearby_predictions(
    latitude: float,
    longitude: float,
    radius: float = 10.0,
    email: str = Depends(verify_token)
):
    """Get all predictions within radius (km) of location"""
    
    user_location = {"latitude": latitude, "longitude": longitude}
    nearby_predictions = []
    
    for loc_key, prediction in PREDICTIONS_DB.items():
        pred_location = prediction["location"]
        distance = calculate_distance(user_location, pred_location)
        
        if distance <= radius:
            prediction_copy = prediction.copy()
            prediction_copy["distance_km"] = round(distance, 2)
            nearby_predictions.append(prediction_copy)
    
    # Sort by distance
    nearby_predictions.sort(key=lambda x: x["distance_km"])
    
    return {
        "count": len(nearby_predictions),
        "predictions": nearby_predictions
    }

# ============================================================================
# API ENDPOINTS - ZONES
# ============================================================================

@app.get("/api/zones/all")
async def get_all_zones(email: str = Depends(verify_token)):
    """Get all monitored zones"""
    return {
        "count": len(ZONES_DB),
        "zones": list(ZONES_DB.values())
    }

@app.get("/api/zones/{zone_id}")
async def get_zone_details(zone_id: str, email: str = Depends(verify_token)):
    """Get detailed information about specific zone"""
    if zone_id not in ZONES_DB:
        raise HTTPException(status_code=404, detail="Zone not found")
    
    return ZONES_DB[zone_id]

# ============================================================================
# API ENDPOINTS - COMMUNITY REPORTS
# ============================================================================

@app.post("/api/reports/submit", status_code=status.HTTP_201_CREATED)
async def submit_community_report(
    report: CommunityReport,
    background_tasks: BackgroundTasks,
    email: str = Depends(verify_token)
):
    """Submit community report about water quality or sanitation issues"""
    
    report_data = report.dict()
    report_data["id"] = len(REPORTS_DB) + 1
    report_data["reporter_email"] = email
    report_data["submitted_at"] = datetime.now().isoformat()
    report_data["status"] = "pending"
    
    REPORTS_DB.append(report_data)
    
    # Trigger risk recalculation in background
    # background_tasks.add_task(recalculate_zone_risk, report.location)
    
    return {
        "message": "Report submitted successfully",
        "report_id": report_data["id"],
        "status": "pending"
    }

@app.get("/api/reports/recent")
async def get_recent_reports(
    limit: int = 10,
    email: str = Depends(verify_token)
):
    """Get recent community reports"""
    sorted_reports = sorted(
        REPORTS_DB,
        key=lambda x: x["submitted_at"],
        reverse=True
    )
    return {
        "count": len(sorted_reports[:limit]),
        "reports": sorted_reports[:limit]
    }

# ============================================================================
# API ENDPOINTS - HEALTH LOGGING
# ============================================================================

@app.post("/api/health/log-symptom", status_code=status.HTTP_201_CREATED)
async def log_health_symptom(
    symptom: HealthSymptom,
    email: str = Depends(verify_token)
):
    """Log health symptom for surveillance"""
    
    symptom_data = symptom.dict()
    symptom_data["id"] = len(HEALTH_LOGS_DB) + 1
    symptom_data["reporter_email"] = email
    symptom_data["logged_at"] = datetime.now().isoformat()
    
    HEALTH_LOGS_DB.append(symptom_data)
    
    return {
        "message": "Symptom logged successfully",
        "log_id": symptom_data["id"]
    }

@app.get("/api/health/my-logs")
async def get_user_health_logs(email: str = Depends(verify_token)):
    """Get user's health symptom logs"""
    user_logs = [log for log in HEALTH_LOGS_DB if log["reporter_email"] == email]
    return {
        "count": len(user_logs),
        "logs": user_logs
    }

# ============================================================================
# API ENDPOINTS - ALERTS
# ============================================================================

@app.get("/api/alerts/active")
async def get_active_alerts(
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    email: str = Depends(verify_token)
):
    """Get active alerts, optionally filtered by location"""
    
    active_alerts = [
        alert for alert in ALERTS_DB
        if alert.get("expires_at") is None or 
        datetime.fromisoformat(alert["expires_at"]) > datetime.now()
    ]
    
    if latitude and longitude:
        user_location = {"latitude": latitude, "longitude": longitude}
        relevant_alerts = []
        
        for alert in active_alerts:
            alert_location = alert["location"]
            distance = calculate_distance(user_location, alert_location)
            
            if distance <= alert["affected_radius"]:
                alert_copy = alert.copy()
                alert_copy["distance_km"] = round(distance, 2)
                relevant_alerts.append(alert_copy)
        
        return {
            "count": len(relevant_alerts),
            "alerts": relevant_alerts
        }
    
    return {
        "count": len(active_alerts),
        "alerts": active_alerts
    }

# ============================================================================
# API ENDPOINTS - STATISTICS & ANALYTICS
# ============================================================================

@app.get("/api/analytics/dashboard")
async def get_dashboard_analytics(email: str = Depends(verify_token)):
    """Get dashboard analytics and statistics"""
    
    # Calculate statistics
    total_zones = len(ZONES_DB)
    high_risk_zones = sum(1 for z in ZONES_DB.values() if z["risk_level"] in [RiskLevel.HIGH, RiskLevel.CRITICAL])
    total_reports = len(REPORTS_DB)
    recent_reports = len([r for r in REPORTS_DB if datetime.fromisoformat(r["submitted_at"]) > datetime.now() - timedelta(days=7)])
    
    return {
        "overview": {
            "total_zones": total_zones,
            "high_risk_zones": high_risk_zones,
            "total_reports": total_reports,
            "recent_reports_7d": recent_reports,
            "active_alerts": len(ALERTS_DB)
        },
        "risk_distribution": {
            "low": sum(1 for z in ZONES_DB.values() if z["risk_level"] == RiskLevel.LOW),
            "moderate": sum(1 for z in ZONES_DB.values() if z["risk_level"] == RiskLevel.MODERATE),
            "high": sum(1 for z in ZONES_DB.values() if z["risk_level"] == RiskLevel.HIGH),
            "critical": sum(1 for z in ZONES_DB.values() if z["risk_level"] == RiskLevel.CRITICAL)
        }
    }

# ============================================================================
# ROOT & HEALTH CHECK
# ============================================================================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "ClimaTrack API",
        "version": "1.0.0",
        "description": "Predictive Waterborne Disease Risk Monitoring System",
        "docs": "/api/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected",  # In production, check actual DB
        "ml_model": "loaded"  # In production, check actual model
    }

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )