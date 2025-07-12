# Heart Disease Prediction using Logistic Regression (From Scratch)

This project implements **Logistic Regression from scratch using NumPy** to predict heart disease from patient clinical data. No ML libraries (like `sklearn` for modeling) were used for the training — the core logic is custom-built to understand how logistic regression works under the hood.

---

## Overview

The dataset used is the **Heart Disease Cleveland Dataset**, a classic medical dataset that contains various patient features like age, cholesterol, blood pressure, etc., along with a target label indicating heart disease presence.

This project is great for:
- Understanding the mathematics behind logistic regression
- Building ML algorithms without pre-built libraries
- Practicing model evaluation and tuning

---

##Dataset

- Source: [Kaggle - Heart Disease Cleveland UCI](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
- Total samples: ~300 (after cleaning)
- Features include:
  - `age`, `sex`, `chol`, `trtbps` (resting BP), `thalachh` (max HR), etc.
- Target:
  - `condition = 0` → no heart disease
  - `condition > 0` → heart disease (binarized to 1)

---

## ⚙️ Model Implementation

Implemented using:
- `numpy` for numerical operations
- `sklearn` only for preprocessing and dataset handling (no model fitting)

### Key Components:
- Sigmoid function
- Gradient computation
- Gradient descent optimization
- Threshold-based binary prediction
- Accuracy evaluation

---

## 🔧 How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/heart-disease-logistic-numpy.git
   cd heart-disease-logistic-numpy
2. install Dependencies
   ```bash
   pip install numpy pandas scikit-learn matplotlib
3. Download the dataset from Kaggle and place heart_cleveland_upload.csv in the root directory.
4. Run the script
   ```bash
   python main.py

##Results

| **Metric**           | **Score** |
|----------------------|-----------|
| Training Accuracy    | 86.47%    |
| Test Accuracy        | 81.11%    |

---

## Next Steps

- ✅ Add **loss vs iteration** plot  
- ✅ Add **comparison with sklearn’s logistic regression**  
- 🔜 Add **L2 regularization** support  
- 🔜 Add **mini-batch gradient descent**  
- 🔜 Add **ROC curve and AUC** metrics

---

##Learning Outcomes

- Built **logistic regression from the ground up**
- Gained **intuition on how features and gradients interact**
- Understood **hyperparameter tuning** (learning rate `alpha`, number of iterations)
- Applied the model to **real-world health data**

---

## 📌 Credits

- 📊 **Dataset**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- 🎓 **Inspiration**: Andrew Ng’s ML course + `scikit-learn` workflows

---

## 📬 Contact

**Yashasvi Kumar Tiwari**  
📧 yashasvikumartiwari@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/yashasvi-kumar-tiwari/)  
🔗 [GitHub](https://github.com/matrixton02)
