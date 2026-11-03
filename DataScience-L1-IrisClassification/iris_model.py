import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path = r'C:\Users\prasa\OIBSIP\DataScience-L1-IrisClassification\Iris.csv'
df = pd.read_csv(path)
print(df.head())
print('\nColumns:', list(df.columns))

label_col = 'Species'
if label_col not in df.columns:
    raise KeyError(f"Expected label column '{label_col}' not found. Available columns: {list(df.columns)}")

X = df.drop(columns=[label_col])
y = df[label_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f'Accuracy: {acc:.4f}')
print(classification_report(y_test, pred))
cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.xticks(range(len(model.classes_)), model.classes_)
plt.yticks(range(len(model.classes_)), model.classes_)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.tight_layout()
plt.savefig(r'C:\Users\prasa\OIBSIP\DataScience-L1-IrisClassification\output\confusion_matrix.png', dpi=150)
plt.close()
with open(r'C:\Users\prasa\OIBSIP\DataScience-L1-IrisClassification\output\model_metrics.txt', 'w', encoding='utf-8') as f:
    f.write('Accuracy: ' + format(acc, '.4f') + '\n')
    f.write(classification_report(y_test, pred))
