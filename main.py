import pandas as pd
import joblib

# Load the dataset
df = pd.read_csv('dataset_phishing.csv')
print(df.head())
print(df['status'].value_counts())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Label encoding
df['status'] = df['status'].map({'phishing': 1, 'legitimate': 0})
df = df.drop_duplicates()

# Features and labels
X = df['url']
y = df['status']

# Vectorize URLs
vectorizer = TfidfVectorizer(max_features=3000)
X_vectors = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

test_url = ["http://secure-paypal-login.com/login"]  # You can change this
test_vector = vectorizer.transform(test_url)
prediction = model.predict(test_vector)

if prediction[0] == 1:
    print("⚠️ Phishing site detected!")
else:
    print("✅ Legitimate site.")


#saveitforlater #loading
joblib.dump(model, 'phishing_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
#saving
model = joblib.load('phishing_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


