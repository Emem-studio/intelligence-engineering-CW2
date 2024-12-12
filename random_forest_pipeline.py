import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(directory_path):
    """Load and preprocess the dataset."""
    combined_data = pd.DataFrame()
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            temp_data = pd.read_csv(file_path)
            combined_data = pd.concat([combined_data, temp_data], ignore_index=True)

    combined_data_cleaned = combined_data.dropna()

    label_mapping = {
        1: "Sitting",
        6: "Walking",
        7: "Running",
        8: "Lying Down"
    }

    combined_data_cleaned['activity'] = combined_data_cleaned['label'].map(label_mapping)
    combined_data_filtered = combined_data_cleaned[combined_data_cleaned['activity'].notnull()]

    sensor_columns = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    X = combined_data_filtered[sensor_columns]
    y = combined_data_filtered['activity'].astype('category')

    return X, y

def feature_selection(X_train, y_train, X_test, k=4):
    """Select top k features."""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_reduced = selector.fit_transform(X_train, y_train)
    X_test_reduced = selector.transform(X_test)
    return X_train_reduced, X_test_reduced

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate a Random Forest model."""
    rf_model = RandomForestClassifier(n_estimators=10, max_depth=10, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', xticklabels=y_test.cat.categories, yticklabels=y_test.cat.categories)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    """Main execution workflow."""
    directory_path = r'C:\Users\AFO\Documents\har70plus'

    # Load and preprocess data
    X, y = load_and_preprocess_data(directory_path)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Feature selection
    X_train_reduced, X_test_reduced = feature_selection(X_train, y_train, X_test)

    # Train and evaluate model
    train_random_forest(X_train_reduced, y_train, X_test_reduced, y_test)

if __name__ == "__main__":
    main()
