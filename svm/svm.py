import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv(r'C:\Users\Mihalis\Desktop\NCSR AI\deep learning project\AI-Generated-Images-vs-Real-Images-Classification\data\features.csv')

X = df.drop('label', axis = 1).values
y = df['label'].values

svm_classifier = SVC(kernel='linear', probability=True)

# Set up k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

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
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train
    svm_classifier.fit(X_train, y_train)
    
    # Pred
    y_pred = svm_classifier.predict(X_test)
    y_prob = svm_classifier.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))
    conf_matrix += confusion_matrix(y_test, y_pred)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label=f'ROC fold {len(aucs)} (AUC = {roc_auc:.2f})')

# Plot mean ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=0.8)
plt.plot([0, 1], [0, 1], linestyle='--', color='r', lw=2, alpha=0.8)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Plot confusion matrix
labels = ['Real', 'AI-generated']
sns.heatmap(conf_matrix, annot=True, fmt ='g', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

# Print metrics
print(f"Mean Accuracy: {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
print(f"Mean Precision: {np.mean(precisions):.2f} ± {np.std(precisions):.2f}")
print(f"Mean Recall: {np.mean(recalls):.2f} ± {np.std(recalls):.2f}")
print(f"Mean F1 Score: {np.mean(f1_scores):.2f} ± {np.std(f1_scores):.2f}")

print("Confusion Matrix:")
print(conf_matrix)
