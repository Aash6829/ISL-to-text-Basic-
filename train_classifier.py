import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ===== Load dataset =====
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# ===== Split dataset =====
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42
)

# ===== Train model =====
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(x_train, y_train)

# ===== Evaluate =====
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)
print(f"✅ {score * 100:.2f}% accuracy")

# ===== Save trained model =====
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("💾 Model saved as 'model.p'")


# ===== Detailed metrics =====
print("\n📊 Classification Report:")
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# ===== Create accuracy table =====
accuracy_table = pd.DataFrame(report).transpose()
print("\nAccuracy Table:")
print(accuracy_table)

# ===== Plot confusion matrix =====
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(labels),
            yticklabels=np.unique(labels))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
