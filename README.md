# Phishing URL Detection using Machine Learning

This project detects phishing websites based on their URLs using machine learning. It uses the **TF-IDF Vectorization** technique to transform URLs into numerical features and applies a **Logistic Regression** model to classify URLs as phishing or legitimate.

## Project Overview

1. **Data**: The dataset consists of a collection of URLs, labeled as `phishing` or `legitimate`. The data is cleaned and processed using the **pandas** library.
2. **Machine Learning Model**: The **Logistic Regression** model is trained using the **scikit-learn** library. 
3. **URL Testing**: After training the model, it can predict whether a URL is a phishing site or a legitimate one.
4. **Model Saving**: The model is saved as a `.pkl` file for future use.

## Requirements

- Python 3.x
- pandas
- scikit-learn
- joblib
- matplotlib (optional for visualizations)

## Steps to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PhishingURLDetector.git
