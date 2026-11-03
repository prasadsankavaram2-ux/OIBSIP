# Iris Flower Classification — OASIS INFOBYTE Data Science Internship

**Author:** Prasad Sankavaram  
**Track:** Data Science  
**Task:** Task 1 — Iris Flower Classification  
**Level:** Beginner

## Project overview
This beginner machine-learning project classifies an Iris flower as **Setosa**, **Versicolor**, or **Virginica** using four measurements: sepal length, sepal width, petal length, and petal width.

## Objectives
- Load the built-in Iris dataset from `sklearn.datasets.load_iris()`.
- Inspect dataset dimensions, data types, missing values, and descriptive statistics.
- Explore feature distributions using a pairplot and feature-wise boxplots.
- Discuss which measurements are most discriminative.
- Split the data into training and test sets using an 80/20 stratified split.
- Train Logistic Regression and Random Forest classifiers.
- Compare accuracy, precision, recall, F1-score, and confusion matrices.
- Select the best model based on held-out test accuracy.

## Tech stack
Python, pandas, scikit-learn, matplotlib, seaborn, Jupyter Notebook.

## Repository structure
```text
OIBSIP/
└── DataScience-L1-IrisClassification/
    ├── iris_classification.ipynb
    ├── iris_classification.py
    ├── requirements.txt
    ├── README.md
    ├── PROJECT_REPORT.pdf
    ├── LINKEDIN_PRESENTATION.md
    └── outputs/
        ├── pairplot.png
        ├── feature_boxplots.png
        ├── best_model_confusion_matrix.png
        ├── feature_importance.png
        ├── model_comparison.csv
        └── evaluation_summary.json
```

## Setup
Python 3.9 or newer is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the project
Run the Python script:
```bash
python iris_classification.py
```
Or open and run `iris_classification.ipynb` cell by cell:
```bash
jupyter notebook
```

The script creates plots and evaluation files in the `outputs/` directory. The notebook contains the same workflow with explanations.

## Dataset
The project uses scikit-learn's built-in Iris dataset. It contains 150 samples, 4 numeric measurements, and 3 species classes. No separate dataset download is needed.

## Evaluation
Two classifiers are evaluated on the same stratified 20% test set:
1. Logistic Regression (with feature standardization).
2. Random Forest (200 trees).

The notebook/script reports accuracy and a per-class precision, recall, and F1-score, plus confusion matrices. The best model is selected from the measured test accuracy. Exact metrics are generated when you run the notebook or script; they are intentionally not hard-coded.

## Feature interpretation
Petal length and petal width are typically more discriminative for separating the species than sepal measurements. Confirm this observation using the pairplot, boxplots, and Random Forest feature-importance chart produced by your run.

## Limitations
- The Iris dataset is small and clean compared with many real-world datasets.
- A single train/test split can produce a different score from another split.
- The result should not be interpreted as a guarantee of performance on every real-world flower measurement.

## Internship submission checklist
- [ ] Review and run the complete notebook.
- [ ] Verify generated plots and metrics.
- [ ] Add your own GitHub repository URL and LinkedIn profile details where appropriate.
- [ ] Push this folder to the single repository named **OIBSIP** using the exact folder name `DataScience-L1-IrisClassification`.
- [ ] Record an end-to-end demo video. First 2 seconds must show full name, assigned track, and task title.
- [ ] Publish the demo video on LinkedIn, tag Oasis Infobyte, and include `#oasisinfobyte` and `#datascience`.
- [ ] Comment substantively on at least two other interns' demo videos.
- [ ] Submit the OIBSIP repository link through the task submission form shared by the internship team.

## Author
Prasad Sankavaram
