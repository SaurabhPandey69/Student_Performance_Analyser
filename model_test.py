import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import json

def load_models():
    """Load trained models"""
    classifier = joblib.load('decision_tree_classifier.pkl')
    regressor = joblib.load('decision_tree_regressor.pkl')
    le_performance = joblib.load('label_encoder_performance.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    
    return classifier, regressor, le_performance, label_encoders, feature_cols

def load_test_data():
    """Load and prepare test data"""
    df = pd.read_csv('student_data.csv')
    
    # Encode categorical variables
    categorical_cols = ['Extracurricular', 'Parent_Education', 'Internet_Access', 'Grade']
    
    for col in categorical_cols:
        le = joblib.load('label_encoders.pkl')[col]
        df[col] = le.transform(df[col])
    
    feature_cols = ['Study_Hours', 'Attendance', 'Previous_Score', 'Sleep_Hours', 
                    'Extracurricular', 'Parent_Education', 'Internet_Access']
    
    X = df[feature_cols]
    y_true_class = df['Performance']
    y_true_reg = df['Score']
    
    return X, y_true_class, y_true_reg, df

def test_models():
    """Test both models on the entire dataset"""
    print("\n" + "="*60)
    print("MODEL TESTING RESULTS")
    print("="*60)
    
    # Load models and data
    classifier, regressor, le_performance, label_encoders, feature_cols = load_models()
    X, y_true_class, y_true_reg, df = load_test_data()
    
    # Test classification model
    print("\n📊 CLASSIFICATION MODEL TESTING")
    print("-" * 40)
    y_pred_class_encoded = classifier.predict(X)
    y_pred_class = le_performance.inverse_transform(y_pred_class_encoded)
    
    # Convert true labels to original format
    y_true_class_original = df['Performance']
    
    accuracy = accuracy_score(y_true_class_original, y_pred_class)
    print(f"Overall Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_true_class_original, y_pred_class))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true_class_original, y_pred_class))
    
    # Test regression model
    print("\n📈 REGRESSION MODEL TESTING")
    print("-" * 40)
    y_pred_reg = regressor.predict(X)
    r2 = r2_score(y_true_reg, y_pred_reg)
    mse = mean_squared_error(y_true_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    
    # Sample predictions
    print("\n🔍 SAMPLE PREDICTIONS (First 10 students)")
    print("-" * 60)
    sample_results = []
    for i in range(min(10, len(X))):
        actual_class = y_true_class_original.iloc[i]
        predicted_class = y_pred_class[i]
        actual_score = y_true_reg.iloc[i]
        predicted_score = y_pred_reg[i]
        
        result = {
            'Student': i+1,
            'Actual_Performance': actual_class,
            'Predicted_Performance': predicted_class,
            'Actual_Score': round(actual_score, 2),
            'Predicted_Score': round(predicted_score, 2),
            'Score_Difference': round(abs(actual_score - predicted_score), 2)
        }
        sample_results.append(result)
        
        print(f"Student {i+1}: Actual={actual_class}({actual_score:.1f}) → "
              f"Predicted={predicted_class}({predicted_score:.1f}) "
              f"[Diff: {abs(actual_score-predicted_score):.1f}]")
    
    # Feature importance
    print("\n🌟 FEATURE IMPORTANCE")
    print("-" * 40)
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': classifier.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for idx, row in feature_importance.iterrows():
        print(f"{row['Feature']:20s}: {row['Importance']:.4f} ({row['Importance']*100:.1f}%)")
    
    return sample_results

def test_single_student():
    """Test prediction for a single student"""
    print("\n" + "="*60)
    print("SINGLE STUDENT PREDICTION TEST")
    print("="*60)
    
    # Load models
    classifier, regressor, le_performance, label_encoders, feature_cols = load_models()
    
    # Test student data
    test_student = {
        'Study_Hours': 8,
        'Attendance': 95,
        'Previous_Score': 85,
        'Sleep_Hours': 8,
        'Extracurricular': 'Yes',
        'Parent_Education': 'Master',
        'Internet_Access': 'Yes'
    }
    
    print("\n📝 Test Student Data:")
    for key, value in test_student.items():
        print(f"  {key}: {value}")
    
    # Prepare features
    features = []
    for col in feature_cols:
        value = test_student[col]
        if col in label_encoders:
            value = label_encoders[col].transform([value])[0]
        features.append(value)
    
    # Make predictions
    predicted_score = regressor.predict([features])[0]
    predicted_class_encoded = classifier.predict([features])[0]
    predicted_class = le_performance.inverse_transform([predicted_class_encoded])[0]
    
    print(f"\n🎯 Predictions:")
    print(f"  Predicted Score: {predicted_score:.2f}")
    print(f"  Predicted Performance: {predicted_class}")
    
    # Suggestions based on prediction
    print(f"\n💡 Suggestions for Improvement:")
    if predicted_score < 60:
        print("  • Increase study hours (aim for 6-8 hours daily)")
        print("  • Improve attendance to 90%+")
        print("  • Ensure 7-8 hours of sleep")
    elif predicted_score < 75:
        print("  • Focus on weak subjects")
        print("  • Maintain consistent study schedule")
        print("  • Consider additional tutoring if needed")
    else:
        print("  • Great performance! Maintain consistency")
        print("  • Consider taking advanced courses")
        print("  • Help peers to reinforce learning")

if __name__ == "__main__":
    # Run tests
    sample_results = test_models()
    test_single_student()
    
    print("\n✅ Model testing completed successfully!")