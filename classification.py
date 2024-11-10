import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

wine_dataset = pd.read_csv("/Users/rash/Downloads/wine+quality/winequality-white.csv",sep=';')

binning = [0,4,7,10]

given_labels = ['Low Quality', 'Average Quality', 'High Quality']
wine_dataset['quality'] = pd.cut(wine_dataset['quality'], bins=binning, labels=given_labels, include_lowest=True)

print(wine_dataset.head())

X = wine_dataset.drop('quality', axis=1)
Y = wine_dataset['quality']

X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)
X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, Y_train)

# Validate the model on validation set
knn_valid_predictions = knn_classifier.predict(X_valid_scaled)
knn_valid_accuracy = accuracy_score(Y_valid, knn_valid_predictions)
knn_valid_precision = precision_score(Y_valid, knn_valid_predictions, average='macro')
knn_valid_recall = recall_score(Y_valid, knn_valid_predictions, average='macro')

print("Validation Metrics for k-Nearest Neighbors Classifier:")
print(f"Accuracy: {knn_valid_accuracy:.2f}")
print(f"Precision: {knn_valid_precision:.2f}")
print(f"Recall: {knn_valid_recall:.2f}")


# Evaluate on the test set
knn_test_predictions = knn_classifier.predict(X_test_scaled)
knn_test_accuracy = accuracy_score(Y_test, knn_test_predictions)
knn_test_precision = precision_score(Y_test, knn_test_predictions, average='macro')
knn_test_recall = recall_score(Y_test, knn_test_predictions, average='macro')

print("Test Metrics for k-Nearest Neighbors Classifier:")
print(f"Accuracy: {knn_test_accuracy:.2f}")
print(f"Precision: {knn_test_precision:.2f}")
print(f"Recall: {knn_test_recall:.2f}")


#IMPLEMENTING RANDOM FOREST CLASSIFIER FOR THE PREDICTION

# Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train_scaled, Y_train)

# Validate the model on validation set
rf_valid_predictions = random_forest.predict(X_valid_scaled)
rf_valid_accuracy = accuracy_score(Y_valid, rf_valid_predictions)
rf_valid_precision = precision_score(Y_valid, rf_valid_predictions, average='macro')
rf_valid_recall = recall_score(Y_valid, rf_valid_predictions, average='macro')


print("Validation Metrics for Random Forest Classifier:")
print(f"Accuracy: {rf_valid_accuracy:.2f}")
print(f"Precision (Macro Average): {rf_valid_precision:.2f}")
print(f"Recall: {rf_valid_recall:.2f}")



# Evaluate on the test set
rf_test_predictions = random_forest.predict(X_test_scaled)
rf_test_accuracy = accuracy_score(Y_test, rf_test_predictions)
rf_test_precision = precision_score(Y_test, rf_test_predictions, average='macro')
rf_test_recall = recall_score(Y_test, rf_test_predictions, average='macro')


print("Test Metrics for Random Forest Classifier:")
print(f"Accuracy: {rf_test_accuracy:.2f}")
print(f"Precision: {rf_test_precision:.2f}")
print(f"Recall: {rf_test_recall:.2f}")



