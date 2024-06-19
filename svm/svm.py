import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

start_time_train = time.time()

# Load training set
train_df = pd.read_csv(r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\data\features.csv')

X_train = train_df.drop('label', axis=1).values
y_train = train_df['label'].values

# Load test set
test_df = pd.read_csv(r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\data\test_features.csv')

X_test = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

# Load extra test set
my_test_df = pd.read_csv(r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\data\my_test_features.csv')

X_my_test = my_test_df.drop('label', axis=1).values
y_my_test = my_test_df['label'].values

svm_classifier = SVC(kernel='linear', probability=True)

# Set up k-folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Metric lists
accuracies = []
precisions = []
recalls = []
f1_scores = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
conf_matrix = np.zeros((2, 2))

# K-folds
for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

    # Train
    svm_classifier.fit(X_train_fold, y_train_fold)
    
    # Validate
    y_val_pred = svm_classifier.predict(X_val_fold)
    y_val_prob = svm_classifier.predict_proba(X_val_fold)[:, 1]
    
    # Calculate metrics for validation set
    accuracies.append(accuracy_score(y_val_fold, y_val_pred))
    precisions.append(precision_score(y_val_fold, y_val_pred))
    recalls.append(recall_score(y_val_fold, y_val_pred))
    f1_scores.append(f1_score(y_val_fold, y_val_pred))
    conf_matrix += confusion_matrix(y_val_fold, y_val_pred)
    
    # Calculate ROC curve and AUC for validation set
    fpr, tpr, _ = roc_curve(y_val_fold, y_val_prob)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label=f'ROC fold {len(aucs)} (AUC = {roc_auc:.2f})')

# Time
end_time_train = time.time()
elapsed_time_train = end_time_train - start_time_train
hours_train, rem_train = divmod(elapsed_time_train, 3600)
minutes_train, seconds_train = divmod(rem_train, 60)
print(f"Training for 5 folds completed in :  {int(hours_train)}h {int(minutes_train)}m {int(seconds_train)}s")

# Plot mean ROC curve for validation set
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=0.8)
plt.plot([0, 1], [0, 1], linestyle='--', color='r', lw=2, alpha=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Validation Set)')
plt.legend(loc='lower right')
plt.show()

# Plot confusion matrix for validation set
labels = ['AI-generated', 'Real']
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix (Validation Set)')
plt.show()

# Print metrics for validation set
print(f"Mean Accuracy: {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
print(f"Mean Precision: {np.mean(precisions):.2f} ± {np.std(precisions):.2f}")
print(f"Mean Recall: {np.mean(recalls):.2f} ± {np.std(recalls):.2f}")
print(f"Mean F1 Score: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")
print("Confusion Matrix (Validation Set):")
print(conf_matrix)

# Train on the whole training set
svm_classifier.fit(X_train, y_train)

# Evaluate on the test set
y_test_pred = svm_classifier.predict(X_test)
y_test_prob = svm_classifier.predict_proba(X_test)[:, 1]

# Calculate metrics for test set
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_conf_matrix = confusion_matrix(y_test, y_test_pred)

# Calculate ROC curve and AUC for test set
test_fpr, test_tpr, _ = roc_curve(y_test, y_test_prob)
test_roc_auc = auc(test_fpr, test_tpr)
plt.plot(test_fpr, test_tpr, color='b', label=f'Test ROC (AUC = {test_roc_auc:.2f})', lw=2, alpha=0.8)
plt.plot([0, 1], [0, 1], linestyle='--', color='r', lw=2, alpha=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Test Set)')
plt.legend(loc='lower right')
plt.show()

# Plot confusion matrix for test set
sns.heatmap(test_conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix (Test Set)')
plt.show()

# Print metrics for test set
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Test Precision: {test_precision:.2f}")
print(f"Test Recall: {test_recall:.2f}")
print(f"Test F1 Score: {test_f1:.2f}")
print("Confusion Matrix (Test Set):")
print(test_conf_matrix)

# Evaluate on the extra test set
y_my_test_pred = svm_classifier.predict(X_my_test)
y_my_test_prob = svm_classifier.predict_proba(X_my_test)[:, 1]

# Calculate metrics for extra test set
my_test_accuracy = accuracy_score(y_my_test, y_my_test_pred)
my_test_precision = precision_score(y_my_test, y_my_test_pred)
my_test_recall = recall_score(y_my_test, y_my_test_pred)
my_test_f1 = f1_score(y_my_test, y_my_test_pred)
my_test_conf_matrix = confusion_matrix(y_my_test, y_my_test_pred)

# Calculate ROC curve and AUC for extra test set
my_test_fpr, my_test_tpr, _ = roc_curve(y_my_test, y_my_test_prob)
my_test_roc_auc = auc(my_test_fpr, my_test_tpr)
plt.plot(my_test_fpr, my_test_tpr, color='b', label=f'My Test ROC (AUC = {my_test_roc_auc:.2f})', lw=2, alpha=0.8)
plt.plot([0, 1], [0, 1], linestyle='--', color='r', lw=2, alpha=0.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (My Test Set)')
plt.legend(loc='lower right')
plt.show()

# Plot confusion matrix for extra test set
sns.heatmap(my_test_conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix (My Test Set)')
plt.show()

# Print metrics for extra test set
print(f"My Test Accuracy: {my_test_accuracy:.2f}")
print(f"My Test Precision: {my_test_precision:.2f}")
print(f"My Test Recall: {my_test_recall:.2f}")
print(f"My Test F1 Score: {my_test_f1:.2f}")
print("Confusion Matrix (My Test Set):")
print(my_test_conf_matrix)
