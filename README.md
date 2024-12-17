# **Heart Disease Prediction Using Machine Learning**

## **Overview**
This project predicts the likelihood of heart disease using machine learning techniques. The dataset was sourced from the **UCI Machine Learning Repository** and includes key health metrics like age, cholesterol, blood pressure, and more.  
Three models—**Logistic Regression**, **Naive Bayes**, and **K-Nearest Neighbors (KNN)**—were trained and evaluated to compare their performance in predicting heart disease.

## **Dataset Information**
- **Source**: UCI Machine Learning Repository (Heart Disease Dataset)
- **Number of Samples**: 303  
- **Number of Features**: 14  
- **Target Variable**:  
  - `1`: Presence of Heart Disease  
  - `0`: Absence of Heart Disease

### Key Features:
- **age**: Age of the patient  
- **cp**: Chest pain type (categorical)  
- **chol**: Serum cholesterol level in mg/dl  
- **thalach**: Maximum heart rate achieved  
- **oldpeak**: ST depression induced by exercise  
- **slope**: Slope of the peak exercise ST segment  

## **Steps to Run the Project**
1. Clone or download this repository.  
2. Install the required libraries using:
   ```bash
   pip install -r requirements.txt
   ```
3. If you don't have **Jupyter Notebook**, install it along with the necessary kernel:
   ```bash
   pip install notebook ipykernel
   ```
4. Launch **Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
5. Open the notebook file (`heart_disease_prediction.ipynb`) and run the cells sequentially.

## **Data Preprocessing**
- Missing values were handled by imputing the mean.  
- Features were standardized using `StandardScaler` from `scikit-learn`.  
- No significant class imbalance was found, so SMOTE or resampling was not applied.


## **Model Comparison**
Three machine learning models were implemented:
1. **Logistic Regression**
2. **Naive Bayes**
3. **K-Nearest Neighbors (KNN)**

### **Model Performance**  
| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | **85%**  | 0.84      | 0.85   | 0.84     |
| Naive Bayes          | 80%      | 0.78      | 0.80   | 0.79     |
| KNN                  | 78%      | 0.76      | 0.78   | 0.77     |

**Observations**:  
- Logistic Regression performed the best with an accuracy of **85%**.  
- Naive Bayes showed competitive performance but slightly lower accuracy.  
- KNN had the lowest accuracy, possibly due to sensitivity to noise and the small dataset.

---

## **Visualizations and Insights**
1. **Feature Correlation Heatmap**:  
   - Strong correlations were observed for features like **oldpeak**, **thal**, and **slope**.  
   - Weak or negligible correlations were noted for **chol** and **fbs**.  

2. **Model Comparison Visualization**:  
   - A bar chart comparing the accuracy of all three models shows Logistic Regression as the top performer.  

---

## **Conclusion**
Logistic Regression outperformed Naive Bayes and KNN, likely due to its ability to handle linear relationships within the dataset. This project demonstrates that machine learning can be a powerful tool for early detection of heart disease, aiding healthcare professionals in timely intervention.

---

## **Acknowledgments**
- **UCI Machine Learning Repository** for providing the dataset.
