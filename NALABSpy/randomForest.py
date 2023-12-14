# Import the required libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Create an imputer object with a strategy to replace missing values with 0
imputer_zero = SimpleImputer(strategy='constant', fill_value=0)

# Load the annotated FIDE dataset
dataset = pd.read_csv('./requirementsNALABSData.csv')

# Preprocess the dataset by selecting the quality attributes
selected_attributes = ['ID', 'Subjectivity Detected', 'Low Readability', 'Not a Requirement']

X = dataset[selected_attributes]
y = dataset['Not a Requirement']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Impute missing values with 0
X_train_imputed = imputer_zero.fit_transform(X_train)
X_test_imputed = imputer_zero.transform(X_test)

y_train_imputed = imputer_zero.fit_transform(y_train.values.reshape(-1, 1))
y_test_imputed = imputer_zero.transform(y_test.values.reshape(-1, 1))

print("Imputed training set:", X_train_imputed)


# Create a Random Forest classifier model
model = RandomForestClassifier(random_state=42)

# Train the model using the imputed training set
model.fit(X_train_imputed, y_train_imputed.ravel())

# Evaluate the model's performance using the imputed testing set
y_pred = model.predict(X_test_imputed)
accuracy = accuracy_score(y_test_imputed, y_pred)
print("Accuracy:", accuracy)

