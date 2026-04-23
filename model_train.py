import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    df = pd.read_csv('student_data.csv')
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['Extracurricular', 'Parent_Education', 'Internet_Access', 'Grade']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Features for classification (performance prediction)
    feature_cols = ['Study_Hours', 'Attendance', 'Previous_Score', 'Sleep_Hours', 
                    'Extracurricular', 'Parent_Education', 'Internet_Access']
    
    X = df[feature_cols]
    y_class = df['Performance']  # For classification
    y_reg = df['Score']  # For regression
    
    # Encode target variable for classification
    le_performance = LabelEncoder()
    y_class_encoded = le_performance.fit_transform(y_class)
    
    return X, y_class_encoded, y_reg, le_performance, label_encoders, feature_cols

def train_classification_model(X, y):
    """Train decision tree classifier"""
    print("\n" + "="*50)
    print("TRAINING DECISION TREE CLASSIFIER")
    print("="*50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    clf = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        criterion='gini'
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.2%}")
    
    return clf

def train_regression_model(X, y):
    """Train decision tree regressor"""
    print("\n" + "="*50)
    print("TRAINING DECISION TREE REGRESSOR")
    print("="*50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    reg = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    reg.fit(X_train, y_train)
    
    # Evaluate
    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\nModel R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    
    return reg

def save_models(classifier, regressor, le_performance, label_encoders, feature_cols):
    """Save trained models and encoders"""
    joblib.dump(classifier, 'decision_tree_classifier.pkl')
    joblib.dump(regressor, 'decision_tree_regressor.pkl')
    joblib.dump(le_performance, 'label_encoder_performance.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(feature_cols, 'feature_columns.pkl')
    
    print("\n" + "="*50)
    print("MODELS SAVED SUCCESSFULLY")
    print("="*50)
    print("Saved files:")
    print("  - decision_tree_classifier.pkl")
    print("  - decision_tree_regressor.pkl")
    print("  - label_encoder_performance.pkl")
    print("  - label_encoders.pkl")
    print("  - feature_columns.pkl")

def visualize_tree(classifier, feature_cols):
    """Visualize the decision tree"""
    plt.figure(figsize=(20, 10))
    plot_tree(classifier, 
              feature_names=feature_cols,
              class_names=['High', 'Medium', 'Low'],
              filled=True, 
              rounded=True,
              fontsize=8)
    plt.title('Student Performance Decision Tree', fontsize=16)
    plt.tight_layout()
    plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
    print("\nDecision tree visualization saved as 'decision_tree_visualization.png'")
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    X, y_class, y_reg, le_performance, label_encoders, feature_cols = load_and_preprocess_data()
    
    # Train models
    classifier = train_classification_model(X, y_class)
    regressor = train_regression_model(X, y_reg)
    
    # Save models
    save_models(classifier, regressor, le_performance, label_encoders, feature_cols)
    
    # Visualize decision tree
    visualize_tree(classifier, feature_cols)
    
    print("\n✅ Model training completed successfully!")