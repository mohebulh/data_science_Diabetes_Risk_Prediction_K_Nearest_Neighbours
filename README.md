# Diabetes Risk Prediction with K-Nearest Neighbours

**An end-to-end data science project that cleans a real clinical dataset, explores the main risk factors for diabetes, and builds K-Nearest Neighbours (KNN) models that predict whether a patient is diabetic from routine health measurements.**

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter

---

## Why This Matters

Diabetes affects millions of people in the UK and costs the NHS billions of pounds a year. Many people at high risk don't know it. If a few routine measurements, such as blood glucose, blood pressure and BMI, can flag at-risk patients early, doctors can intervene before serious complications develop.

This project asks three questions, framed as hypotheses from a team of doctors and researchers:

| # | Hypothesis | Predictors used |
|---|---|---|
| 1 | Which levels of glucose, blood pressure and BMI put a patient at high risk of diabetes? | Glucose, BloodPressure, BMI |
| 2 | Which age groups are most vulnerable to diabetes? | Age |
| 3 | How well does family history predict diabetes risk? | DiabetesPedigreeFunction |

## Dataset

- **Source:** [Diabetes – Intermediate Dataset](https://www.kaggle.com/datasets/samira1992/diabetes-intermediate-dataset) on Kaggle, originally from the US National Institute of Diabetes and Digestive and Kidney Diseases.
- **Population:** female patients aged 21 and over.
- **Size after cleaning:** 531 patients, 8 features and 1 target.

| Feature | Description |
|---|---|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration two hours into an oral glucose tolerance test |
| `BloodPressure` | Diastolic blood pressure (mmHg) |
| `SkinThickness` | Triceps skinfold thickness (mm) |
| `Insulin` | Two-hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (kg/m²) |
| `DiabetesPedigreeFunction` | Score summarising family history of diabetes |
| `Age` | Age in years |
| `Outcome` | **Target:** 1 = diabetic, 0 = not diabetic |

**Class balance:** 354 non-diabetic (66.7%) and 177 diabetic (33.3%).

## Workflow

```
Raw data → Cleaning → Exploratory analysis → Feature selection → KNN training and tuning → Evaluation → Interactive prediction
```

### 1. Data cleaning
Medically impossible values were treated as missing data and removed:
- **Blood pressure of 0.** A living person cannot have a diastolic pressure of zero.
- **BMI of 0.** Every person has a BMI above zero.
- **Glucose of 0.** An invalid reading.
- **Skin thickness of 0 mm.** Not physically possible.

Then, in Python:
- checked for and removed duplicate rows,
- confirmed there were no null values with `info()`,
- reviewed summary statistics (mean, standard deviation, minimum, quartiles, maximum) with `describe()`.

### 2. Exploratory data analysis
- Count plot and pie chart of the outcome classes.
- Glucose, blood pressure and BMI each split into ten value ranges, showing the **percentage of diabetic patients in each range**.
- Age distribution of diabetic patients.
- Diabetic rate for each Diabetes Pedigree Function band (0–2.5 in steps of 0.25).

### 3. Modelling
- **Algorithm:** K-Nearest Neighbours classifier (scikit-learn).
- **Hyperparameter tuning:** tested k values from 1 to 73 in steps of 6 and plotted accuracy against k to choose the best value.
- **Train/test split:** 95% training, 5% test, with a fixed random seed for reproducibility.
- **Interactive prediction:** each model accepts new patient values typed in by the user and returns a diabetic or non-diabetic prediction.

## Results

| Hypothesis | Model | Test accuracy |
|---|---|---|
| 1. Glucose + blood pressure + BMI | KNN, k = 13 (tuned) | **92.6%** |
| 2. Age only | KNN, k = 13 | 70.4% |
| 3. Diabetes Pedigree Function only | KNN, k = 19 | 59.3% |

### Key findings
- **Glucose, blood pressure and BMI together are by far the strongest predictors.** The share of diabetic patients rises clearly as each of these values increases.
- **Patients with a higher Diabetes Pedigree Function were more often diabetic**, which supports family history as a risk factor.
- **Age or family history on its own is a weak predictor.** Those single-feature models performed much worse than the combined model, so they work best as supporting signals alongside clinical measurements.



Planned improvements:
- [ ] Use an 80/20 split with **stratified k-fold cross-validation** for tuning, and keep the test set for final evaluation only.
- [ ] **Scale features** (StandardScaler), because KNN is distance-based and the features have very different ranges.
- [ ] Report **precision, recall, F1, a confusion matrix and ROC-AUC**. Recall matters most here, because missing a diabetic patient is the costliest error.
- [ ] Build the workflow as a scikit-learn **Pipeline** and compare KNN with logistic regression and random forest.
- [ ] Impute zero values in `Insulin` instead of keeping them.

## Repository Contents

| File | Description |
|---|---|
| [`Diabetes_risk_prediction_data_science.ipynb`](./Diabetes_risk_prediction_data_science.ipynb) | Main notebook: cleaning, analysis, visualisation and KNN models |
| [`Diabetes.csv`](./Diabetes.csv) | Cleaned dataset (531 rows) |
| [`Diabetes_risk_prediction_Hypothesis.pdf`](./Diabetes_risk_prediction_Hypothesis.pdf) | Hypothesis report: problem definition, hypotheses and manual pre-processing |
| [`Diabetes_risk_prediction_Data_Science_report.pdf`](./Diabetes_risk_prediction_Data_Science_report.pdf) | Full project report: method, visualisations, results and reflection |

## How to Run

```bash
git clone https://github.com/mohebulh/data_science_Diabetes_Risk_Prediction_K_Nearest_Neighbours.git
cd data_science_Diabetes_Risk_Prediction_K_Nearest_Neighbours
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook Diabetes_risk_prediction_data_science.ipynb
```

Run the cells in order. The prediction cells ask you to type in patient values.

## Skills Demonstrated

Data cleaning and validation · exploratory data analysis · data visualisation (Matplotlib, Seaborn) · feature selection · supervised machine learning · K-Nearest Neighbours · hyperparameter tuning · model evaluation · hypothesis-driven analysis · Python · pandas · NumPy · scikit-learn · Jupyter · technical report writing

---
