# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load processed datasets
print("Loading preprocessed datasets...")
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# -------------------------------
# MODEL 1: Gradient Boosting (XGBoost)
# -------------------------------

print("\nTraining XGBoost Classifier...")
xgb_model = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# XGBoost predictions and evaluation
y_pred_xgb = xgb_model.predict(X_test)
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))

# Plot confusion matrix (XGBoost)
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('xgb_confusion_matrix.png')
plt.show()

# -------------------------------
# MODEL 2: Deep Neural Network (TensorFlow/Keras)
# -------------------------------

print("\nTraining Deep Neural Network...")
dnn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile model
dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train DNN
history = dnn_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=1024,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate the DNN model
y_pred_dnn_prob = dnn_model.predict(X_test).ravel()
y_pred_dnn = (y_pred_dnn_prob >= 0.5).astype(int)

print("\nDeep Neural Network Classification Report:")
print(classification_report(y_test, y_pred_dnn))

# Plot confusion matrix (DNN)
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test, y_pred_dnn), annot=True, fmt='d', cmap='Greens')
plt.title('DNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('dnn_confusion_matrix.png')
plt.show()

# ROC Curves for both models
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_model.predict_proba(X_test)[:,1])
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

fpr_dnn, tpr_dnn, _ = roc_curve(y_test, y_pred_dnn_prob)
roc_auc_dnn = auc(fpr_dnn, tpr_dnn)

plt.figure(figsize=(8,6))
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
plt.plot(fpr_dnn, tpr_dnn, label=f'DNN (AUC = {roc_auc_dnn:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.savefig('roc_curve_comparison.png')
plt.show()

print("\nModel training and evaluation completed successfully.")
