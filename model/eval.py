import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, auc, accuracy_score

class PaymentClassifier(nn.Module):
    def __init__(self, input_dim):
        super(PaymentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3) 
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  
        return x

def classify_payment(row):
    if row['total_amount_paid_per_line'] == row['total_charge_per_line']:
        return "Paid in Full"
    elif row['total_amount_paid_per_line'] == 0:
        return "Not Paid"
    else:
        return "Partially Paid"
    
import pandas as pd
df = pd.read_parquet("processed_data.parquet")

df['payment_status'] = df.apply(classify_payment, axis=1)
label_encoder = LabelEncoder()
df['payment_status'] = label_encoder.fit_transform(df['payment_status']) 

df = df.fillna(0)

X = df.drop(columns=['payment_status', 'total_amount_paid_per_line']).values
y = df['payment_status'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_dim = X.shape[1]
model = PaymentClassifier(input_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

model.eval()
y_true, y_pred, y_scores = [], [], []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        probabilities = torch.softmax(outputs, dim=1)[:, 1]  # Probabilities for class 1 (partially paid)
        _, predicted = torch.max(outputs, 1)  # Predicted class labels
        
        y_true.extend(batch_y.numpy())
        y_pred.extend(predicted.numpy())
        y_scores.extend(probabilities.numpy())  # Store predicted probabilities for PR-AUC

# Compute Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Compute PR-AUC
precision, recall, _ = precision_recall_curve(y_true, y_scores, pos_label=1)  
pr_auc = auc(recall, precision)  
print(f"PR-AUC: {pr_auc:.4f}")

# Compute Recall @ 95% Precision
threshold_idx = next((i for i, p in enumerate(precision) if p >= 0.95), -1)
recall_at_95 = recall[threshold_idx] if threshold_idx != -1 else 0  
print(f"Recall @ 95% Precision: {recall_at_95:.4f}")

cm = confusion_matrix(y_true, y_pred)
class_labels = list(label_encoder.classes_)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))
