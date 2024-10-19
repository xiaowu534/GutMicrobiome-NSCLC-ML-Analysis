import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from collections import Counter

# Set a fixed random seed for reproducibility
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def build_and_evaluate_models(response_column, models_to_train, feature_data_path, metadata_path, results_directory, correlation_method='pearson', top_feature_percentage=100):
    """
    Constructs and assesses multiple machine learning models for predicting patient immunotherapy responses.

    Parameters:
    - response_column (str): The name of the column in the metadata that contains the clinical response labels.
    - models_to_train (list): A list of model names to be trained and evaluated.
    - feature_data_path (str): File path to the CSV containing microbial features.
    - metadata_path (str): File path to the CSV containing metadata, including clinical responses.
    - results_directory (str): Directory where the output files will be saved.
    - correlation_method (str): Method for calculating feature correlation ('pearson', 'spearman', 'kendall').
    - top_feature_percentage (float): Percentage of top features to select based on correlation.

    Returns:
    - performance_results (dict): A dictionary containing ROC AUC scores and confusion matrices for each model.
    """
    # Load microbial feature data and metadata
    features_df = pd.read_csv(feature_data_path)
    metadata_df = pd.read_csv(metadata_path)
    
    # Select necessary metadata columns
    metadata_df = metadata_df[['SampleID', response_column]]
    
    # Merge feature data with metadata
    merged_data = pd.merge(features_df, metadata_df, on='SampleID')
    merged_data.dropna(inplace=True)
    
    # Calculate correlation matrix for feature selection
    if correlation_method in ['pearson', 'spearman', 'kendall']:
        correlation_matrix = merged_data.iloc[:, 1:-1].corr(method=correlation_method)
    else:
        raise ValueError("Unsupported correlation method. Choose 'pearson', 'spearman', or 'kendall'.")
    
    # Select top features based on correlation
    abs_corr = correlation_matrix.abs()
    average_corr = abs_corr.mean(axis=0)
    selected_features = average_corr.sort_values(ascending=False).head(int(len(average_corr) * (top_feature_percentage / 100))).index
    
    X = merged_data[selected_features]
    y = LabelEncoder().fit_transform(merged_data[response_column])
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    # Balance the training data by oversampling minority classes
    class_counts = Counter(y_train)
    max_class_count = max(class_counts.values())
    
    X_train_balanced = pd.DataFrame()
    y_train_balanced = np.array([])
    
    for cls, count in class_counts.items():
        indices = np.where(y_train == cls)[0]
        repeats = max_class_count // count
        remainder = max_class_count % count
        X_cls = pd.concat([X_train.iloc[indices]] * repeats + [X_train.iloc[indices][:remainder]])
        y_cls = np.tile(y_train[indices], repeats + 1)[:max_class_count]
        X_train_balanced = pd.concat([X_train_balanced, X_cls])
        y_train_balanced = np.concatenate([y_train_balanced, y_cls])
    
    # Normalize the feature data
    scaler = StandardScaler()
    X_train_balanced = scaler.fit_transform(X_train_balanced)
    X_test = scaler.transform(X_test)
    
    # Define the machine learning models to be used
    model_definitions = {
        'Logistic_Regression': LogisticRegression(C=0.1, solver='saga', max_iter=1000),
        'Linear_SVM': SVC(kernel='linear', C=0.5, probability=True),
        'Naive_Bayes': GaussianNB(var_smoothing=1e-9),
        'Radial_SVM': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
        'Decision_Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=2, min_samples_leaf=10),
        'Random_Forest': RandomForestClassifier(n_estimators=200, max_depth=1000),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, learning_rate=0.05, max_depth=6),
        'Neural_Network': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=100)
    }
    
    # Select the models specified for training
    chosen_models = {name: model for name, model in model_definitions.items() if name in models_to_train}
    performance_results = {}
    
    for model_name, model in chosen_models.items():
        # Train the model
        model.fit(X_train_balanced, y_train_balanced)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)
            y_proba = y_scores[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        
        # Compute ROC AUC score and confusion matrix
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        cm = confusion_matrix(y_test, y_pred)
        performance_results[model_name] = {'ROC_AUC': roc_auc, 'Confusion_Matrix': cm}
        
        # Plot and save ROC curves for each class
        plt.figure()
        for i in range(len(np.unique(y_test))):
            fpr, tpr, _ = roc_curve(y_test == i, y_proba)
            plt.plot(fpr, tpr, label=f'Class {i} ROC curve')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves for {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(results_directory, f'{model_name}_roc_curve.png'))
        plt.close()
        
        # Save ROC data to Excel
        roc_data = pd.DataFrame({
            'False_Positive_Rate': fpr,
            'True_Positive_Rate': tpr
        })
        roc_data.to_excel(os.path.join(results_directory, f'{model_name}_roc_data.xlsx'), index=False)
        
        # Print the confusion matrix
        print(f'Confusion Matrix for {model_name}:\n{cm}\n')
        
    return performance_results

def main():
    # Define the configurations for different datasets
    configurations = [
        {
            'feature_data_path': 'features_genus.csv',
            'metadata_path': 'metadata.csv',
            'response_column': 'Clinical_Response',
            'results_directory': 'results_genus'
        },
        {
            'feature_data_path': 'features_species.csv',
            'metadata_path': 'metadata.csv',
            'response_column': 'Clinical_Response',
            'results_directory': 'results_species'
        }
    ]
    
    # List of models to train
    models_to_train = [
        'Logistic_Regression', 'Linear_SVM', 'Naive_Bayes', 'Radial_SVM',
        'Decision_Tree', 'Random_Forest', 'XGBoost', 'Neural_Network'
    ]
    correlation_method = 'pearson'
    top_feature_percentage = 95
    
    # Run the model training and evaluation for each configuration
    for config in configurations:
        results_directory = config['results_directory']
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)
        
        print(f"Processing dataset: {config['feature_data_path']} with metadata: {config['metadata_path']}")
        performance_results = build_and_evaluate_models(
            response_column=config['response_column'],
            models_to_train=models_to_train,
            feature_data_path=config['feature_data_path'],
            metadata_path=config['metadata_path'],
            results_directory=results_directory,
            correlation_method=correlation_method,
            top_feature_percentage=top_feature_percentage
        )
        print(f"Results for {results_directory}: {performance_results}\n")

if __name__ == "__main__":
    main()
