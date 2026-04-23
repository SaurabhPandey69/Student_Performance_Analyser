import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_student_data(n_samples=1000):
    """
    Generate synthetic student performance data
    """
    data = []
    
    # Define possible values
    study_hours_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    attendance_options = [60, 65, 70, 75, 80, 85, 90, 95, 100]
    previous_scores_options = [40, 50, 60, 70, 80, 85, 90, 95]
    sleep_hours_options = [4, 5, 6, 7, 8, 9]
    extracurricular_options = ['Yes', 'No']
    parent_education_options = ['High School', 'Bachelor', 'Master', 'PhD']
    internet_access_options = ['Yes', 'No']
    
    for _ in range(n_samples):
        # Generate features
        study_hours = random.choice(study_hours_options)
        attendance = random.choice(attendance_options)
        previous_score = random.choice(previous_scores_options)
        sleep_hours = random.choice(sleep_hours_options)
        extracurricular = random.choice(extracurricular_options)
        parent_education = random.choice(parent_education_options)
        internet_access = random.choice(internet_access_options)
        
        # Calculate performance score based on features
        base_score = 40
        
        # Study hours contribution
        if study_hours >= 8:
            base_score += 25
        elif study_hours >= 6:
            base_score += 18
        elif study_hours >= 4:
            base_score += 10
        else:
            base_score += 2
        
        # Attendance contribution
        if attendance >= 90:
            base_score += 20
        elif attendance >= 80:
            base_score += 12
        elif attendance >= 70:
            base_score += 6
        else:
            base_score += 0
        
        # Previous score contribution
        if previous_score >= 85:
            base_score += 20
        elif previous_score >= 70:
            base_score += 12
        else:
            base_score += 5
        
        # Sleep hours contribution
        if sleep_hours >= 8:
            base_score += 8
        elif sleep_hours >= 6:
            base_score += 4
        else:
            base_score -= 5
        
        # Extracurricular adjustment
        if extracurricular == 'Yes':
            base_score -= 3  # Slight negative impact if too many activities
        
        # Parent education contribution
        if parent_education == 'PhD':
            base_score += 10
        elif parent_education == 'Master':
            base_score += 7
        elif parent_education == 'Bachelor':
            base_score += 4
        
        # Internet access contribution
        if internet_access == 'Yes':
            base_score += 5
        
        # Add some random noise
        noise = np.random.normal(0, 5)
        final_score = base_score + noise
        
        # Ensure score is between 0 and 100
        final_score = max(0, min(100, final_score))
        
        # Determine grade based on score
        if final_score >= 90:
            grade = 'A'
        elif final_score >= 80:
            grade = 'B'
        elif final_score >= 70:
            grade = 'C'
        elif final_score >= 60:
            grade = 'D'
        else:
            grade = 'F'
        
        # Determine performance category
        if final_score >= 75:
            performance = 'High'
        elif final_score >= 50:
            performance = 'Medium'
        else:
            performance = 'Low'
        
        data.append({
            'Study_Hours': study_hours,
            'Attendance': attendance,
            'Previous_Score': previous_score,
            'Sleep_Hours': sleep_hours,
            'Extracurricular': extracurricular,
            'Parent_Education': parent_education,
            'Internet_Access': internet_access,
            'Score': round(final_score, 2),
            'Grade': grade,
            'Performance': performance
        })
    
    return pd.DataFrame(data)

def save_data():
    """Generate and save the dataset"""
    print("Generating student performance data...")
    df = generate_student_data(2000)
    
    # Save to CSV
    df.to_csv('student_data.csv', index=False)
    print(f"Dataset saved to student_data.csv")
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nData distribution:")
    print(df['Performance'].value_counts())
    print(df['Grade'].value_counts())
    
    return df

if __name__ == "__main__":
    save_data()