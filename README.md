# 🎓 Student Performance Analyzer

An AI-powered web application that predicts student academic performance using Decision Tree algorithms. The system analyzes multiple factors like study hours, attendance, previous scores, and more to provide accurate predictions and personalized recommendations.

## 🚀 Features

- **Decision Tree Classification**: Predicts performance level (High/Medium/Low)
- **Score Prediction**: Estimates exact percentage scores
- **Interactive Web Interface**: Modern UI with real-time predictions
- **Feature Importance Analysis**: Shows which factors most influence performance
- **Personalized Recommendations**: AI-generated suggestions for improvement
- **RESTful API**: Easy integration with other applications

## 📊 Model Performance

- Classification Accuracy: ~85-90%
- Regression R² Score: ~0.75-0.85
- Cross-validation Score: ~82-87%

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **Machine Learning**: Scikit-learn (Decision Tree)
- **Frontend**: HTML5, CSS3, JavaScript
- **Data Processing**: Pandas, NumPy
- **Model Deployment**: Joblib

## 📁 Project Structure
Student_Performance_Analyser/
│
├── app.py # FastAPI backend server
├── generator.py # Synthetic data generation
├── model_train.py # Decision Tree model training
├── model_test.py # Model evaluation
├── requirements.txt # Python dependencies
├── templates/
│ └── index.html # Frontend interface
└── README.md # Project documentation

text

## 🔧 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Student_Performance_Analyser.git
cd Student_Performance_Analyser
2. Install dependencies

bash
pip install -r requirements.txt
3. Generate synthetic data

bash
python generator.py
4. Train the model

bash
python model_train.py
5. Test the model

bash
python model_test.py
6. Run the web application

bash
uvicorn app:app --reload
7. Open in browser

text
http://localhost:8000
📖 Usage

Enter student information in the web form:

Study hours per day
Attendance percentage
Previous exam score
Sleep hours
Extracurricular activities
Parent education level
Internet access
Click "Predict Performance"
View results:

Predicted score percentage
Performance category (High/Medium/Low)
Confidence level
Personalized recommendations
Key factors analysis
🎯 Features Importance

The model identifies these key factors (in order of importance):

Study Hours
Previous Score
Attendance
Sleep Hours
Parent Education
Internet Access
Extracurricular Activities
📈 API Endpoints

Endpoint	Method	Description
/	GET	Web interface
/api/predict	POST	Make prediction
/api/feature-importance	GET	Get feature importance
/api/health	GET	Health check
API Example

bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "study_hours": 8,
    "attendance": 95,
    "previous_score": 85,
    "sleep_hours": 8,
    "extracurricular": "No",
    "parent_education": "Master",
    "internet_access": "Yes"
  }'
🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author

Your Name - [Your GitHub Profile]

🙏 Acknowledgments

Scikit-learn for machine learning algorithms
FastAPI for the awesome web framework
All contributors and users of this project
📧 Contact

For questions or feedback, please open an issue on GitHub.
