
# OkCupid Neural Network (PyTorch, Jupyter Notebook)

This project is based on the **Codecademy OkCupid Machine Learning Project**. The goal is to train and evaluate a **PyTorch neural network** to predict a user’s **body type** from a subset of demographic and lifestyle features, all inside a **Jupyter Notebook**.  

---

## 📌 Project Overview  

This is a **supervised classification task**:  

- **Input features**:  
  - `age`  
  - `drinks`  
  - `drugs`  
  - `smokes`  
  - `diet`  

- **Target variable**:  
  - `body_type` (categorical, multi-class)  

The project includes preprocessing, exploratory analysis, model training in PyTorch, and evaluation with classification metrics.  

---

## 🗂 Dataset  

The dataset (provided by Codecademy) is anonymized and only for educational use.  
- **Numerical**: `age`  
- **Categorical**: `drinks`, `drugs`, `smokes`, `diet`  
- **Target**: `body_type`  

---

## ⚙️ Tech Stack  

- **Python 3**  
- **Jupyter Notebook**  
- **Pandas / NumPy** → preprocessing  
- **scikit-learn** → preprocessing utilities & baseline models  
- **PyTorch** → neural network training  

---

## 🚀 Project Workflow  

1. **Data Preprocessing**  
   - Handle missing values  
   - Encode categorical variables  
   - Normalize numerical values (`age`)  

2. **Exploratory Data Analysis (EDA)**  
   - Distribution of target (`body_type`)  
   - Feature relationships (e.g., drinks vs. body type)  

3. **Modeling**  
   - Baseline models (logistic regression, decision tree)  
   - PyTorch neural network:  
     - Input = encoded features  
     - Hidden layers = ReLU activations, dropout  
     - Output = softmax over `body_type` classes  

4. **Training & Evaluation**  
   - Loss: CrossEntropyLoss  
   - Optimizer: Adam  
   - Metrics: Accuracy, Precision, Recall, F1  

---

## 📊 Results  

- Baselines provide a starting reference  
- The PyTorch model learns nonlinear feature interactions and improves classification  
- Performance varies depending on class imbalance in `body_type`  

*(Insert actual accuracy/F1/confusion matrix results here once available.)*  

---

## ▶️ How to Run  

1. Clone the repo:  
   ```bash
   git clone https://github.com/yourusername/body-type-ML.git
   cd body-type-ML
   ```  

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

3. Launch Jupyter Notebook:  
   ```bash
   jupyter notebook
   ```  

4. Open and run:  
   ```
   date-a-scientist.ipynb
   ```  

---

## 📌 Requirements  

```
pandas
numpy
scikit-learn
torch
jupyter
```  

---

## 📖 Credits  

- **Codecademy**: OkCupid Machine Learning Project inspiration  
- **OkCupid dataset**: Educational anonymized profile data  
