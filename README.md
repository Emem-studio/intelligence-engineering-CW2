# intelligence-engineering-CW2
# Random Forest Activity Classification Pipeline

## Project Description

This project implements a pipeline for activity classification using sensor data from wearable devices. The primary objective is to classify human activities such as `Walking`, `Running`, `Lying Down`, and `Sitting` based on features extracted from accelerometer sensors placed on different body parts. The project uses a Random Forest Classifier to achieve accurate activity recognition.

## Features

- Preprocessing pipeline for handling missing values, mapping labels, and filtering activities.
- Feature selection using `SelectKBest`.
- Random Forest Classifier for activity prediction.
- Hyperparameter tuning using `GridSearchCV`.
- Model evaluation with metrics such as accuracy, classification report, and confusion matrix.

## Libraries Used

The following libraries are used in this project:

- `os`
- `pandas`
- `sklearn`
  - `train_test_split`
  - `RandomForestClassifier`
  - `classification_report`, `confusion_matrix`, `accuracy_score`
  - `GridSearchCV`
  - `SelectKBest`, `f_classif`
- `matplotlib`
- `seaborn`

## Dataset

The dataset contains sensor readings from wearable devices. Each data entry includes measurements such as `back_x`, `back_y`, `back_z`, `thigh_x`, `thigh_y`, and `thigh_z` and their corresponding activity labels.

## Instructions to Set Up and Run the Project

### Prerequisites

1. Python 3.8 or higher.
2. The required libraries listed in `requirements.txt`.

### Setup Steps

1. Clone this repository:

   ```bash
   git clone https://github.com/Emem-studio/intelligence-engineering-CW2.git
   cd intelligence-engineering-CW2
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your dataset files are in the appropriate directory (`har70plus`) as specified in the script.

4. Run the main script:

   ```bash
   python random_forest_pipeline.py
   ```

### Output

- The script outputs accuracy, classification reports, and confusion matrices for the classification task.
- Best hyperparameters from `GridSearchCV` will also be displayed.

## Dataset Details

- **Features**:
  - `back_x`, `back_y`, `back_z`
  - `thigh_x`, `thigh_y`, `thigh_z`
- **Activities**:
  - `Walking`
  - `Running`
  - `Lying Down`
  - `Sitting`
- Filtered out: `Standing`

## Model Details

- **Classifier**: Random Forest
- **Hyperparameter Tuning**:
  - `n_estimators`
  - `max_depth`
  - `max_features`
- Balanced class weights to handle class imbalance.

## Results

The final model achieves an accuracy of **\~83.9%** with the following performance metrics:

- **Precision**, **Recall**, and **F1-Score** for each activity.
- Confusion matrix heatmap visualization.

## Automation

- This project integrates with GitHub for CI/CD and automation.
- Set up on Azure for scalable and efficient processing.


