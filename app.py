from fastapi import FastAPI, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI(title="Student Performance Analyzer")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="templates")

# Load models
try:
    classifier = joblib.load('decision_tree_classifier.pkl')
    regressor = joblib.load('decision_tree_regressor.pkl')
    le_performance = joblib.load('label_encoder_performance.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    classifier = None
    regressor = None

class StudentData(BaseModel):
    study_hours: int
    attendance: int
    previous_score: float
    sleep_hours: int
    extracurricular: str
    parent_education: str
    internet_access: str

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main page"""
    with open('templates/index.html', 'r') as f:
        return HTMLResponse(content=f.read())

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "models_loaded": classifier is not None}

@app.post("/api/predict")
async def predict_student_performance(student: StudentData):
    """Predict student performance based on input data"""
    
    if classifier is None or regressor is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Prepare input data
        input_data = {
            'Study_Hours': student.study_hours,
            'Attendance': student.attendance,
            'Previous_Score': student.previous_score,
            'Sleep_Hours': student.sleep_hours,
            'Extracurricular': student.extracurricular,
            'Parent_Education': student.parent_education,
            'Internet_Access': student.internet_access
        }
        
        # Encode categorical variables
        features = []
        for col in feature_cols:
            value = input_data[col]
            if col in label_encoders:
                if value not in label_encoders[col].classes_:
                    # Handle unknown categories
                    value = label_encoders[col].classes_[0]
                value = label_encoders[col].transform([value])[0]
            features.append(value)
        
        # Make predictions
        predicted_score = float(regressor.predict([features])[0])
        predicted_class_encoded = classifier.predict([features])[0]
        predicted_performance = le_performance.inverse_transform([predicted_class_encoded])[0]
        
        # Calculate confidence (based on decision tree probability)
        probabilities = classifier.predict_proba([features])[0]
        confidence = float(max(probabilities))
        
        # Generate recommendations
        recommendations = generate_recommendations(student, predicted_score)
        
        return JSONResponse(content={
            "success": True,
            "predicted_score": round(predicted_score, 2),
            "predicted_performance": predicted_performance,
            "confidence": round(confidence, 2),
            "recommendations": recommendations,
            "input_data": input_data
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/feature-importance")
async def get_feature_importance():
    """Get feature importance from the decision tree"""
    if classifier is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    importance = classifier.feature_importances_
    features = feature_cols
    
    feature_importance = [
        {"feature": feat, "importance": float(imp)} 
        for feat, imp in zip(features, importance)
    ]
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    return JSONResponse(content={"feature_importance": feature_importance})

def generate_recommendations(student: StudentData, predicted_score: float):
    """Generate personalized recommendations"""
    recommendations = []
    
    if student.study_hours < 6:
        recommendations.append({
            "area": "Study Hours",
            "current": f"{student.study_hours} hours/day",
            "suggestion": "Increase study hours to 6-8 hours per day for better results",
            "impact": "High"
        })
    
    if student.attendance < 85:
        recommendations.append({
            "area": "Attendance",
            "current": f"{student.attendance}%",
            "suggestion": "Improve attendance to 90%+ to stay consistent with curriculum",
            "impact": "High"
        })
    
    if student.sleep_hours < 7:
        recommendations.append({
            "area": "Sleep Hours",
            "current": f"{student.sleep_hours} hours",
            "suggestion": "Get at least 7-8 hours of sleep for better concentration",
            "impact": "Medium"
        })
    
    if student.previous_score < 70:
        recommendations.append({
            "area": "Previous Performance",
            "current": f"{student.previous_score}%",
            "suggestion": "Focus on strengthening fundamentals and seek extra help",
            "impact": "High"
        })
    
    if predicted_score < 60:
        recommendations.append({
            "area": "Overall Performance",
            "current": f"Predicted: {predicted_score:.1f}%",
            "suggestion": "Consider creating a strict study schedule and joining study groups",
            "impact": "Critical"
        })
    elif predicted_score < 75 and not recommendations:
        recommendations.append({
            "area": "Performance Improvement",
            "current": f"Predicted: {predicted_score:.1f}%",
            "suggestion": "Good foundation! Focus on consistent revision and practice tests",
            "impact": "Medium"
        })
    elif predicted_score >= 75:
        recommendations.append({
            "area": "Excellence Track",
            "current": f"Predicted: {predicted_score:.1f}%",
            "suggestion": "Excellent! Consider challenging yourself with advanced topics",
            "impact": "Low"
        })
    
    return recommendations

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)