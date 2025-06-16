# Fraudulent Job Posting Detection

# Project Overview
This project aims to detect fraudulent job postings using machine learning techniques. It processes job listing data, trains a model, and provides a prediction dashboard using Streamlit.

## Key Features
- Preprocessing of real-world job post data
- Logistic Regression for fraud detection
- TF-IDF Vectorizer for text feature extraction
- Interactive Streamlit dashboard
- Visualization (pie chart & histogram)

##  Technologies Used
- Python
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Streamlit


#  Data Science Workflow

# 1. Data Preprocessing
- Handled missing values
- Encoded labels (`fraudulent`)
- Text cleaned and vectorized with TF-IDF

# Model Building
- Trained `LogisticRegression` with class_weight="balanced"
- Evaluated using accuracy, precision, recall, and F1 score

# Prediction
- Saved model using `joblib`
- Predicts on new/unseen job posts

# Dashboard (Streamlit)
- Upload test CSV to see fraud predictions
- Pie chart: Distribution of fraud vs legit
- Histogram: Word count distributions

## Run Instructions

###  Local Setup

bash
pip install -r requirements.txt
streamlit run dashboard.py

# Project Structure

project/
├── dashboard.py
├── train_model.py
├── test_data.csv
├── fraud_job_model.pkl
├── tfidf_vectorizer.pkl
├── README.md


## links
link to streamlit application

https://onlinejobfrauddetection-uji6ofr9orh8gbxwsqjmjg.streamlit.app/

## Google Drive link

https://drive.google.com/drive/folders/1YqNfRBgUeH7BWuFP-aCXGFH4HvohNaUy?usp=sharing
