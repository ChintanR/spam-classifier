# Email/SMS Spam Classifier

An end-to-end machine learning project that classifies messages as spam or ham (not spam), served through a REST API built with FastAPI.

## Problem Statement
Spam messages waste time and can be harmful. The goal is to automatically detect spam while avoiding blocking genuine messages.

## Dataset
SMS Spam Collection (Kaggle / UCI): 5,572 messages. After removing 403 duplicates, 5,169 remained (4,516 ham, 653 spam, about 1:7). The data is imbalanced.

## Approach
1. **EDA:** class imbalance, message length (spam messages are about twice as long on average)
2. **Preprocessing:** lowercasing, punctuation removal, stop-word removal, TF-IDF features
3. **Split:** 80/20 stratified train/test split
4. **Models:** Naive Bayes, Logistic Regression, Random Forest, each in a Scikit-learn Pipeline
5. **Evaluation:** accuracy, precision, recall, F1, confusion matrix, 5-fold cross-validation
6. **Deployment:** best model saved with joblib and served with FastAPI

## Results (test set, 1,034 messages)
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 0.9662 | 0.9898 | 0.7405 | 0.8472 |
| Logistic Regression | 0.9603 | 0.9891 | 0.6947 | 0.8161 |
| Random Forest | 0.9758 | 0.9818 | 0.8244 | 0.8963 |

Always predicting "ham" would give 87.3% accuracy, so accuracy alone is misleading here. Random Forest was chosen for the best F1 and recall while keeping precision high.

## Challenges
- **Class imbalance:** accuracy hides missed spam, so I evaluated with precision, recall and F1.
- **Data leakage:** the split happens before fitting TF-IDF, and the Pipeline keeps the two steps together.
- **Remaining weakness:** about 18% of spam is still missed. Possible improvements are class weights, threshold tuning and keeping symbols like "£" and "!".

## Run Locally
```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs and try `POST /predict`:
```json
{"text": "Congratulations! You have won a free prize, call now to claim"}
```

## Project Structure
```
app/main.py          FastAPI app
src/text_utils.py    text cleaning function
notebooks/           EDA, modelling, evaluation
models/              saved model
data/                dataset (download from Kaggle)
```

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Scikit-learn, FastAPI, Pydantic, Joblib

## My Contribution
Built the whole project individually: data analysis, preprocessing, model training and evaluation, and the API.