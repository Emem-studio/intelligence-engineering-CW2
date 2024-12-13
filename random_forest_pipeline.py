import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def feature_selection(X_train, y_train, X_test, k=4):
    """Select top k features."""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_reduced = selector.fit_transform(X_train, y_train)
    X_test_reduced = selector.transform(X_test)
    
    # Get the indices of selected features
    selected_feature_indices = selector.get_support(indices=True)
    selected_features = X_train.columns[selected_feature_indices]
    print("Selected Features:", list(selected_features))
    
    return X_train_reduced, X_test_reduced, selected_features

def train_random_forest(X_train, y_train, X_test, y_test, n_estimators, max_depth, random_state):
    """Train and evaluate a Random Forest model."""
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        class_weight='balanced', 
        random_state=random_state
    )
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    # Print performance metrics
    print("Accuracy Score:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Get unique classes for confusion matrix labels
    classes = np.unique(np.concatenate((y_test, y_pred)))
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred, labels=classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    return rf_model

def main():
    """Main execution workflow."""
    # Set up argparse for dynamic parameter inputs
    parser = argparse.ArgumentParser(description="Train a Random Forest model with custom parameters.")
    parser.add_argument("--trainingdata", type=str, required=True, help="Path to the training dataset (CSV file).")
    parser.add_argument("--n_estimators", type=int, default=10, help="Number of trees in the forest.")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum depth of the trees.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test data size as a fraction (default: 0.2).")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    # Handle missing arguments
    if not os.path.exists(args.trainingdata):
        raise FileNotFoundError(f"The training data file {args.trainingdata} does not exist.")
    
    # Load data
    print(f"Loading data from {args.trainingdata}...")
    combined_data_filtered = pd.read_csv(args.trainingdata)
    
    # Define features and target
    sensor_columns = ['back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z']
    X = combined_data_filtered[sensor_columns]
    y = combined_data_filtered['activity']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    # Feature selection
    X_train_reduced, X_test_reduced, selected_features = feature_selection(X_train, y_train, X_test)
    
    # Train and evaluate model
    model = train_random_forest(X_train_reduced, y_train, X_test_reduced, y_test, 
                                args.n_estimators, args.max_depth, args.random_state)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
