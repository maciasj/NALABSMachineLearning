import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Create an imputer object with a strategy to replace missing values with 0
imputer_zero = SimpleImputer(strategy='constant', fill_value=0)

# Load the annotated FIDE dataset
dataset = pd.read_csv('./requirementsNALABSData.csv')

# Preprocess the dataset by selecting the quality attributes
selected_attributes = [ 'Subjectivity Detected', 'Low Readability', 'Not a Requirement']

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
# Initialize the Support Vector Machine classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the model
svm_classifier.fit(X_train_imputed, y_train_imputed.ravel())

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_imputed)

# Evaluate accuracy
accuracy = accuracy_score(y_test_imputed, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print("Classification Report:\n", classification_report(y_test_imputed, y_pred))

conf_matrix = confusion_matrix(y_test_imputed, y_pred)
print("Confusion Matrix:\n", conf_matrix)
