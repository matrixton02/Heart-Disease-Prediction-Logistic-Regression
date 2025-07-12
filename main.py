import numpy as np

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def gradient(theta,X,y):
    m=y.size
    return (X.T @ (sigmoid(X @ theta)-y))/m

def gradient_descent(X,y,alpha=0.1,num_iter=100,tol=1e-7):
    X_b=np.c_[np.ones((X.shape[0],1)),X]
    theta=np.zeros(X_b.shape[1])

    for i in range(num_iter):
        grad=gradient(theta,X_b,y)
        theta-=alpha*grad

        if(np.linalg.norm(grad)<tol):
            break
    return theta

def predict_prob(X,theta):
    X_b=np.c_[np.ones((X.shape[0],1)),X]
    return sigmoid(X_b @ theta)

def predict(X,theta,threshold=0.5):
    return (predict_prob(X,theta)>=threshold).astype(int)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("heart_cleveland_upload.csv")

df = df.dropna()  # Drop rows with missing values (can be improved)
df['condition'] = df['condition'].apply(lambda x: 1 if x > 0 else 0)  # Binary classification

# 3. Features and labels
X = df.drop('condition', axis=1).values
y = df['condition'].values

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

theta_hat = gradient_descent(X_train_scaled, y_train, alpha=0.01, num_iter=3000)

# 8. Evaluate
y_pred_train = predict(X_train_scaled, theta_hat)
y_pred_test = predict(X_test_scaled, theta_hat)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:     {test_acc:.4f}")