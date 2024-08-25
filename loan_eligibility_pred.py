import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy import stats

# Load dataset
Data = pd.read_csv('/content/loan_prediction_data.csv')

# Handling missing values by dropping rows with NaNs
Data = Data.dropna()

# Encoding categorical variables if they exist
label_encoders = {}
for column in Data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    Data[column] = label_encoders[column].fit_transform(Data[column])

# Outlier detection and removal using Z-score
z_scores = np.abs(stats.zscore(Data))
Data = Data[(z_scores < 3).all(axis=1)]

# Splitting the dataset into features and target variable
X = Data.iloc[:, 6:11].values
y = Data.iloc[:, 12].values

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Fitting Naive Bayes to the Training Set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Confusion Matrix and Accuracy
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"Confusion Matrix:\n{cm}")
print(f"Accuracy of the model: {accuracy * 100:.2f}%")

# Function to classify new applicants
def classify_applicant():
    a = float(input("Enter applicant's income: "))
    c = float(input("Enter co-applicant's income: "))
    l = int(input("Enter loan amount: "))
    lt = int(input("Enter loan amount tenure (in days): "))
    ch = int(input("Enter 1 for previous loan cleared, 0 for yet to clear: "))
    
    test = np.array([[a, c, l, lt, ch]])
    test = scaler.transform(test)  # Apply the same scaling to the new data
    y_pred = classifier.predict(test)
    
    if y_pred == 1:  # Assuming 1 represents 'Y' and 0 represents 'N'
        print("The Loan applicant is eligible for loan")
    else:
        print("The Loan applicant is not eligible for loan")

# Example usage:
classify_applicant()
