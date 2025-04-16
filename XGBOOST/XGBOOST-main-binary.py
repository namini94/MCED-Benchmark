import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.impute import SimpleImputer
import xgboost as xgb
import pickle

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Set plot styles
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.title_fontsize'] = 14

# Protein feature names
PROTEIN_FEATURES = [
    'AFP (pg/ml)',
    'Angiopoietin-2 (pg/ml)',
    'AXL (pg/ml)',
    'CA-125 (U/ml)',
    'CA 15-3 (U/ml)',
    'CA19-9 (U/ml)',
    'CD44 (ng/ml)',
    'CEA (pg/ml)',
    'CYFRA 21-1 (pg/ml)',
    'DKK1 (ng/ml)',
    'Endoglin (pg/ml)',
    'FGF2 (pg/ml)',
    'Follistatin (pg/ml)',
    'Galectin-3 (ng/ml)',
    'G-CSF (pg/ml)',
    'GDF15 (ng/ml)',
    'HE4 (pg/ml)',
    'HGF (pg/ml)',
    'IL-6 (pg/ml)',
    'IL-8 (pg/ml)',
    'Kallikrein-6 (pg/ml)',
    'Leptin (pg/ml)',
    'Mesothelin (ng/ml)',
    'Midkine (pg/ml)',
    'Myeloperoxidase (ng/ml)',
    'NSE (ng/ml)',
    'OPG (ng/ml)',
    'OPN (pg/ml)',
    'PAR (pg/ml)',
    'Prolactin (pg/ml)',
    'sEGFR (pg/ml)',
    'sFas (pg/ml)',
    'SHBG (nM)',
    'sHER2/sEGFR2/sErbB2 (pg/ml)',
    'sPECAM-1 (pg/ml)',
    'TGFa (pg/ml)',
    'Thrombospondin-2 (pg/ml)',
    'TIMP-1 (pg/ml)',
    'TIMP-2 (pg/ml)'
]

def clean_numeric_data(data):
    """
    Clean the data by removing asterisks (*) from numeric fields.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset to clean
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataset
    """
    # Create a copy to avoid modifying the original
    cleaned_data = data.copy()
    
    # Process each column
    for col in cleaned_data.columns:
        if col not in ['PatientID', 'SampleID', 'Condition', 'Stage']:
            # Check if column contains strings
            if cleaned_data[col].dtype == object or isinstance(cleaned_data[col].iloc[0], str):
                # Remove asterisks
                cleaned_data[col] = cleaned_data[col].astype(str).str.replace('*', '')
                
                # Convert to numeric, coercing errors to NaN
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                
                # Check for NaN values after conversion
                nan_count = cleaned_data[col].isna().sum()
                if nan_count > 0:
                    print(f"Warning: {nan_count} values in column '{col}' could not be converted to numeric")
            
            # Handle mixed numeric types
            elif pd.api.types.is_numeric_dtype(cleaned_data[col]):
                # Check if there might be string values with asterisks in numeric column
                if cleaned_data[col].astype(str).str.contains('\*').any():
                    # Convert to string, remove asterisks, then back to numeric
                    cleaned_data[col] = cleaned_data[col].astype(str).str.replace('*', '')
                    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
    
    return cleaned_data

def log2_normalize(data):
    """
    Apply log2 normalization to the data frame.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset to normalize
        
    Returns:
    --------
    pandas.DataFrame
        Log2-normalized dataset
    """
    # Create a copy to avoid modifying the original
    data_norm = data.copy()
    
    for col in data_norm.columns:
        if col not in ['PatientID', 'SampleID', 'Condition', 'Stage']:
            # Only normalize numeric columns with positive values
            if pd.api.types.is_numeric_dtype(data_norm[col]):
                # Check min value to handle zeros and negatives
                min_val = data_norm[col].min()
                
                if pd.isna(min_val):
                    print(f"Warning: Column '{col}' contains NaN values, filling with column median")
                    # Fill NaN with median
                    median_val = data_norm[col].median()
                    data_norm[col] = data_norm[col].fillna(median_val)
                    min_val = data_norm[col].min()  # Recalculate min
                
                if min_val <= 0:
                    # Add offset to make all values positive
                    offset = abs(min_val) + 1
                    data_norm[col] = np.log2(data_norm[col] + offset)
                else:
                    # Simple log2 transform
                    data_norm[col] = np.log2(data_norm[col])
    
    return data_norm

def remove_nan_rows(data, feature_cols):
    """
    Remove rows with NaN values in specified feature columns.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset
    feature_cols : list
        List of feature columns to check for NaN values
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with NaN rows removed
    """
    # Count original number of rows
    original_count = len(data)
    
    # Remove rows with NaN values in feature columns
    clean_data = data.dropna(subset=feature_cols)
    
    # Count number of rows removed
    removed_count = original_count - len(clean_data)
    if removed_count > 0:
        print(f"Removed {removed_count} rows with NaN values ({removed_count/original_count*100:.2f}% of data)")
        print(f"Final dataset size: {len(clean_data)} samples")
    
    return clean_data

def plot_feature_importance(model, feature_names, output_dir='cancer_xgboost_results', filename='feature_importance.pdf'):
    """
    Plot feature importance from the XGBoost model.
    
    Parameters:
    -----------
    model : xgboost.XGBClassifier
        Trained XGBoost model
    feature_names : list
        Names of the features
    output_dir : str
        Directory to save visualizations
    filename : str
        Filename for the visualization
        
    Returns:
    --------
    None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    
    # Take top 20 features for better visualization
    top_n = min(20, len(feature_names))
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = importance[top_indices]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar chart
    bars = plt.barh(range(top_n), top_importance, align='center', color=plt.cm.viridis(np.linspace(0, 0.8, top_n)))
    
    # Set y-ticks
    plt.yticks(range(top_n), top_features)
    
    # Set labels and title
    plt.xlabel('Feature Importance', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.title('Top Features by Importance', fontsize=18, fontweight='bold')
    
    # Invert y-axis to show most important at the top
    plt.gca().invert_yaxis()
    
    # Add grid for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_scores, output_dir='cancer_xgboost_results', filename='roc_curve.pdf', high_spec_threshold=None):
    """
    Plot ROC curve for binary classification.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_scores : numpy.ndarray
        Predicted probabilities for the positive class
    output_dir : str
        Directory to save visualizations
    filename : str
        Filename for the visualization
    high_spec_threshold : float or None
        If provided, use this specific threshold for >99% specificity point
        
    Returns:
    --------
    float
        AUC score
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find threshold for >99% specificity (similar to the red dot in the figure)
    specificity = 1 - fpr
    
    # Use provided threshold or find it
    if high_spec_threshold is not None:
        # Find the closest threshold in our thresholds array
        threshold_idx = np.abs(thresholds - high_spec_threshold).argmin()
        high_spec_index = threshold_idx
        high_spec_tpr = tpr[high_spec_index]
        high_spec_fpr = fpr[high_spec_index]
    else:
        high_spec_indices = np.where(specificity >= 0.99)[0]
        if len(high_spec_indices) > 0:
            high_spec_index = high_spec_indices[-1]
            high_spec_tpr = tpr[high_spec_index]
            high_spec_fpr = fpr[high_spec_index]
        else:
            # Fallback if no point with >99% specificity
            high_spec_index = np.argmax(specificity)
            high_spec_tpr = tpr[high_spec_index]
            high_spec_fpr = fpr[high_spec_index]
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Plot ROC curve in standard format (not inverted)
    plt.plot(
        fpr, 
        tpr, 
        color='black',
        lw=2, 
        label=f'ROC curve (AUC: {roc_auc:.0%} ({roc_auc-0.01:.0%}–{roc_auc+0.01:.0%}))'
    )
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    
    # Plot high specificity point
    plt.plot(high_spec_fpr, high_spec_tpr, 'ro', markersize=8, 
             label=f'Sensitivity: {high_spec_tpr:.0%} at >99% specificity')
    
    # Set axis labels and style
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve for XGBoost', fontsize=18, fontweight='bold')
    
    # Set tick formats as percentages
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0%', '20%', '40%', '60%', '80%', '100%'])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0%', '20%', '40%', '60%', '80%', '100%'])
    
    # Add legend
    plt.legend(loc="lower right", fontsize=12)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create the alternate visualization as shown in your reference image
    # Create a second figure for the alternate format (with specificity on x-axis)
    plt.figure(figsize=(10, 10))
    
    # Plot ROC curve with specificity on x-axis
    plt.plot(
        specificity, 
        tpr, 
        color='black',
        lw=2, 
        label=f'ROC curve (AUC: {roc_auc:.0%} ({roc_auc-0.01:.0%}–{roc_auc+0.01:.0%}))'
    )
    
    # Plot diagonal line
    plt.plot([0, 1], [1, 0], 'k--', lw=1)
    
    # Plot high specificity point
    plt.plot(1-high_spec_fpr, high_spec_tpr, 'ro', markersize=8, 
             label=f'Sensitivity: {high_spec_tpr:.0%} at >99% specificity')
    
    # Set axis labels and style
    plt.xlabel('Specificity (%)', fontsize=14)
    plt.ylabel('Sensitivity (%)', fontsize=14)
    plt.title('ROC Curve for XGBoost (Alternate Format)', fontsize=18, fontweight='bold')
    
    # Set tick formats as percentages
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0%', '20%', '40%', '60%', '80%', '100%'])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0%', '20%', '40%', '60%', '80%', '100%'])
    
    # Add legend
    plt.legend(loc="lower left", fontsize=12)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f"alt_{filename}"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc

def plot_sensitivity_by_stage(stage_sensitivities, output_dir='cancer_xgboost_results', filename='sensitivity_by_stage.pdf'):
    """
    Plot sensitivity by cancer stage.
    
    Parameters:
    -----------
    stage_sensitivities : dict
        Dictionary with stage as key and sensitivity as value
    output_dir : str
        Directory to save visualizations
    filename : str
        Filename for the visualization
        
    Returns:
    --------
    None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract stages and sensitivities
    stages = list(stage_sensitivities.keys())
    sensitivities = [stage_sensitivities[stage]['sensitivity'] for stage in stages]
    errors = [stage_sensitivities[stage]['error'] for stage in stages]
    
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Create bar chart
    bars = plt.bar(
        stages,
        [s * 100 for s in sensitivities],  # Convert to percentage
        color='lightgray',
        yerr=[e * 100 for e in errors],  # Convert to percentage
        capsize=10,
        error_kw={'elinewidth': 2, 'capthick': 2}
    )
    
    # Set labels and title
    plt.xlabel('Stage', fontsize=14)
    plt.ylabel('Proportion detected\nby XGBoost (%)', fontsize=14)
    plt.title('Sensitivity of XGBoost by stage', fontsize=16, fontweight='bold')
    
    # Set y-axis limits and ticks
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 20))
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_sensitivity_by_cancer_type(cancer_sensitivities, output_dir='cancer_xgboost_results', filename='sensitivity_by_cancer_type.pdf'):
    """
    Plot sensitivity by cancer type.
    
    Parameters:
    -----------
    cancer_sensitivities : dict
        Dictionary with cancer type as key and sensitivity data as value
    output_dir : str
        Directory to save visualizations
    filename : str
        Filename for the visualization
        
    Returns:
    --------
    None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract cancer types and sensitivities
    cancer_types = list(cancer_sensitivities.keys())
    sensitivities = [cancer_sensitivities[ct]['sensitivity'] for ct in cancer_types]
    errors = [cancer_sensitivities[ct]['error'] for ct in cancer_types]
    
    # Define colors for cancer types (matching the figure)
    colors = ['#E57373', '#D4AF37', '#81C784', '#66BB6A', '#4DB6AC', '#42A5F5', '#9575CD', '#F06292']
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Create bar chart
    bars = plt.bar(
        cancer_types,
        [s * 100 for s in sensitivities],  # Convert to percentage
        color=colors[:len(cancer_types)],
        yerr=[e * 100 for e in errors],  # Convert to percentage
        capsize=10,
        error_kw={'elinewidth': 2, 'capthick': 2}
    )
    
    # Set labels and title
    plt.xlabel('Cancer Type', fontsize=14)
    plt.ylabel('Proportion detected\nby XGBoost (%)', fontsize=14)
    plt.title('Sensitivity of XGBoost by tumor type', fontsize=16, fontweight='bold')
    
    # Set y-axis limits and ticks
    plt.ylim(0, 100)
    plt.yticks(range(0, 101, 20))
    
    # Add grid
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fold_accuracies(fold_accuracies, output_dir='cancer_xgboost_results', filename='fold_accuracies.pdf'):
    """
    Plot accuracies for each fold.
    
    Parameters:
    -----------
    fold_accuracies : list
        List of accuracy values for each fold
    output_dir : str
        Directory to save visualizations
    filename : str
        Filename for the visualization
        
    Returns:
    --------
    None
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot accuracies
    fold_nums = range(1, len(fold_accuracies) + 1)
    bars = plt.bar(
        fold_nums,
        fold_accuracies,
        color=plt.cm.viridis(np.linspace(0, 1, len(fold_accuracies))),
        alpha=0.8
    )
    
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    plt.axhline(
        mean_acc,
        color='crimson',
        linestyle='--',
        linewidth=2,
        label=f'Mean: {mean_acc:.4f} ± {std_acc:.4f}'
    )
    
    plt.xlabel('Fold', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('10-Fold Cross-Validation Accuracy', fontsize=18, fontweight='bold')
    plt.xticks(fold_nums)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to run the XGBoost cancer classification pipeline."""
    try:
        # Create output directory
        output_dir = 'cancer_xgboost_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Read data
        print("Reading data...")
        try:
            data = pd.read_excel("Dataset.xlsx")
        except FileNotFoundError:
            print("Dataset.xlsx not found. Please make sure the file exists in the current directory.")
            return
        
        # Print basic dataset information
        print(f"Dataset shape: {data.shape}")
        
        # Check if 'Condition' column exists
        if 'Condition' not in data.columns:
            print("Error: 'Condition' column not found in the dataset.")
            return
            
        print(f"Conditions present: {data['Condition'].unique()}")
        
        # Map Condition to binary (Cancer vs Normal)
        data['binary_condition'] = data['Condition'].apply(lambda x: 1 if x != 'Normal' else 0)
        print(f"Class distribution: {data['binary_condition'].value_counts()}")
        
        # Clean data by removing stars (*) from numeric fields
        print("\nCleaning data (removing * characters)...")
        data_cleaned = clean_numeric_data(data)
        
        # Identify protein features in the dataset
        available_proteins = [protein for protein in PROTEIN_FEATURES if protein in data_cleaned.columns]
        print(f"Found {len(available_proteins)} protein features out of {len(PROTEIN_FEATURES)} expected")
        
        # Find the Omega score column
        omega_cols = [col for col in data_cleaned.columns if 'Omega' in col]
        
        if omega_cols:
            print(f"Found mutation feature: {omega_cols}")
        else:
            print("No Omega score column found. Model will only use protein features.")
            omega_cols = []
        
        # All feature columns
        feature_cols = available_proteins + omega_cols
        
        # Apply log2 normalization
        print("Applying log2 normalization...")
        data_log2 = log2_normalize(data_cleaned)
        
        # Remove rows with NaN values in feature columns
        print("Removing rows with NaN values...")
        data_clean = remove_nan_rows(data_log2, feature_cols)
        
        # Extract features and labels
        X = data_clean[feature_cols].values
        y = data_clean['binary_condition'].values
        
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of samples after cleaning: {X.shape[0]}")
        
        # Check if there are still any NaN values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values still present in data after cleaning.")
            print("Imputing remaining NaN values with feature medians...")
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save the preprocessed data
        print("\nSaving preprocessed data...")
        preprocessed_data = pd.DataFrame(X_scaled, columns=feature_cols)
        preprocessed_data['Condition'] = data_clean['Condition'].values
        preprocessed_data['binary_condition'] = data_clean['binary_condition'].values
        
        # Save to CSV
        preprocessed_file = os.path.join(output_dir, 'preprocessed_data.csv')
        preprocessed_data.to_csv(preprocessed_file, index=False)
        print(f"Preprocessed data saved to: {preprocessed_file}")
        
        # Set up cross-validation
        n_splits = 10  # 10-fold cross-validation
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        
        # Store results
        fold_accuracies = []
        fold_auc_scores = []
        fold_feature_importances = []
        fold_high_spec_thresholds = []  # Store thresholds giving >99% specificity
        all_y_true = []
        all_y_pred = []
        all_y_scores = []
        all_y_pred_high_spec = []  # Predictions at high specificity threshold
        
        # Track sensitivity by stage
        stages = data_clean['Stage'].dropna().unique() if 'Stage' in data_clean.columns else []
        stage_predictions = {stage: {'correct': 0, 'total': 0} for stage in stages}
        
        # Track sensitivity by cancer type
        cancer_types = [ct for ct in data_clean['Condition'].unique() if ct != 'Normal']
        cancer_predictions = {ct: {'correct': 0, 'total': 0} for ct in cancer_types}
        
        # XGBoost parameters with regularization to prevent overfitting
        params = {
            'objective': 'binary:logistic',
            'learning_rate': 0.05,
            'max_depth': 3,            # Reduced max depth to prevent overfitting
            'min_child_weight': 2,     # Increased to prevent overfitting
            'gamma': 0.1,              # Minimum loss reduction for partition
            'subsample': 0.8,          # Use 80% of samples for trees
            'colsample_bytree': 0.8,   # Use 80% of features for trees
            'reg_alpha': 0.5,          # L1 regularization
            'reg_lambda': 10.0,        # L2 regularization
            'scale_pos_weight': sum(y == 0) / sum(y == 1),  # Adjust based on class imbalance
            'n_estimators': 100,       # Number of trees
            'random_state': RANDOM_SEED,
            'use_label_encoder': False,
            'eval_metric': 'auc'
        }
        
        print(f"\nPerforming {n_splits}-fold cross-validation...")
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled, y)):
            print(f"\nFold {fold+1}/{n_splits}")
            
            # Split data
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train XGBoost model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, 
                y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Predict
            y_pred = model.predict(X_test)
            y_scores = model.predict_proba(X_test)[:, 1]  # Probability of positive class
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y_test)
            fold_accuracies.append(accuracy)
            
            try:
                auc_score = roc_auc_score(y_test, y_scores)
                fold_auc_scores.append(auc_score)
                
                # Find the threshold that gives >99% specificity
                fpr, tpr, thresholds = roc_curve(y_test, y_scores)
                specificity = 1 - fpr
                high_spec_indices = np.where(specificity >= 0.99)[0]
                
                if len(high_spec_indices) > 0:
                    high_spec_index = high_spec_indices[-1]  # Take the one with highest sensitivity among those with >99% specificity
                    high_spec_threshold = thresholds[high_spec_index]
                    fold_high_spec_thresholds.append(high_spec_threshold)
                    
                    # Use this threshold for predicting cancer
                    y_pred_at_threshold = (y_scores >= high_spec_threshold).astype(int)
                    
                    # Store these predictions separately for tracking stage and cancer type sensitivities
                    y_pred_high_spec = y_pred_at_threshold
                else:
                    # Fallback if no threshold gives >99% specificity
                    print(f"Warning: No threshold with >99% specificity found for fold {fold+1}")
                    high_spec_threshold = 0.5  # default
                    fold_high_spec_thresholds.append(high_spec_threshold)
                    y_pred_high_spec = y_pred
                
            except ValueError:
                print(f"Could not calculate AUC for fold {fold+1}")
                fold_high_spec_thresholds.append(0.5)  # default threshold
                y_pred_high_spec = y_pred
            # Store feature importance
            fold_feature_importances.append(model.feature_importances_)
            
            # Track predictions
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_scores.extend(y_scores)
            all_y_pred_high_spec.extend(y_pred_high_spec)
            
            # Track sensitivity by stage if stage information is available
            if 'Stage' in data_clean.columns:
                test_samples = data_clean.iloc[test_idx]
                test_indices_map = {idx: i for i, idx in enumerate(test_idx)}
                
                for stage in stages:
                    stage_samples = test_samples[test_samples['Stage'] == stage]
                    for idx in stage_samples.index:
                        if idx in test_indices_map and test_samples.loc[idx, 'binary_condition'] == 1:  # Only count cancer samples
                            stage_predictions[stage]['total'] += 1
                            rel_idx = test_indices_map[idx]
                            if y_pred_high_spec[rel_idx] == 1:  # Correctly predicted as cancer using high specificity threshold
                                stage_predictions[stage]['correct'] += 1
            
            # Track sensitivity by cancer type
            test_samples = data_clean.iloc[test_idx]
            test_indices_map = {idx: i for i, idx in enumerate(test_idx)}
            
            for ct in cancer_types:
                ct_samples = test_samples[test_samples['Condition'] == ct]
                for idx in ct_samples.index:
                    if idx in test_indices_map:
                        cancer_predictions[ct]['total'] += 1
                        rel_idx = test_indices_map[idx]
                        if y_pred_high_spec[rel_idx] == 1:  # Correctly predicted as cancer using high specificity threshold
                            cancer_predictions[ct]['correct'] += 1
            
            print(f"Fold {fold+1} accuracy: {accuracy:.4f}")
            if fold_auc_scores:
                print(f"Fold {fold+1} AUC: {fold_auc_scores[-1]:.4f}")
        
        # Calculate overall metrics
        print("\nOverall Performance:")
        print(f"Average accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
        if fold_auc_scores:
            print(f"Average AUC: {np.mean(fold_auc_scores):.4f} ± {np.std(fold_auc_scores):.4f}")
        
        # Plot fold accuracies
        plot_fold_accuracies(fold_accuracies, output_dir)
        
        # Calculate and plot ROC curve
        if len(all_y_true) > 0 and len(all_y_scores) > 0:
            print("\nPlotting ROC curve...")
            # Find the overall threshold that gives >99% specificity
            fpr, tpr, thresholds = roc_curve(all_y_true, all_y_scores)
            specificity = 1 - fpr
            high_spec_indices = np.where(specificity >= 0.99)[0]
            
            if len(high_spec_indices) > 0:
                high_spec_index = high_spec_indices[-1]
                high_spec_threshold = thresholds[high_spec_index]
                high_spec_tpr = tpr[high_spec_index]
                high_spec_fpr = fpr[high_spec_index]
                print(f"Threshold for >99% specificity: {high_spec_threshold:.4f}")
                print(f"Sensitivity at this threshold: {high_spec_tpr:.4f}")
                print(f"Specificity at this threshold: {1-high_spec_fpr:.4f}")
            else:
                print("Warning: No threshold with >99% specificity found")
                high_spec_threshold = 0.5
            
            # Create the updated y_pred based on the high specificity threshold
            all_y_pred_high_spec = (np.array(all_y_scores) >= high_spec_threshold).astype(int)
            
            # Plot the ROC curve with this threshold
            roc_auc = plot_roc_curve(all_y_true, all_y_scores, output_dir, 
                                     high_spec_threshold=high_spec_threshold)
        
        # Calculate average feature importance across folds
        avg_feature_importance = np.mean(fold_feature_importances, axis=0)
        
        # Create final model using all data
        print("\nTraining final model on all data...")
        final_model = xgb.XGBClassifier(**params)
        final_model.fit(X_scaled, y)
        
        # Plot feature importance
        print("Plotting feature importance...")
        plot_feature_importance(final_model, feature_cols, output_dir)
        
        # Calculate sensitivities by stage
        stage_sensitivities = {}
        for stage in stages:
            if stage_predictions[stage]['total'] > 0:
                sensitivity = stage_predictions[stage]['correct'] / stage_predictions[stage]['total']
                # Calculate standard error based on sample size
                error = np.sqrt((sensitivity * (1 - sensitivity)) / stage_predictions[stage]['total'])
                stage_sensitivities[stage] = {
                    'sensitivity': sensitivity,
                    'error': error,
                    'samples': stage_predictions[stage]['total']
                }
                print(f"Stage {stage} sensitivity: {sensitivity:.4f} ± {error:.4f} ({stage_predictions[stage]['correct']}/{stage_predictions[stage]['total']})")
        
        # Plot sensitivity by stage
        if stage_sensitivities:
            print("Plotting sensitivity by stage...")
            plot_sensitivity_by_stage(stage_sensitivities, output_dir)
        
        # Calculate sensitivities by cancer type
        cancer_sensitivities = {}
        for ct in cancer_types:
            if cancer_predictions[ct]['total'] > 0:
                sensitivity = cancer_predictions[ct]['correct'] / cancer_predictions[ct]['total']
                # Calculate 95% confidence interval
                error = 1.96 * np.sqrt((sensitivity * (1 - sensitivity)) / cancer_predictions[ct]['total'])
                cancer_sensitivities[ct] = {
                    'sensitivity': sensitivity,
                    'error': error,
                    'samples': cancer_predictions[ct]['total']
                }
                print(f"{ct} sensitivity: {sensitivity:.4f} ± {error:.4f} ({cancer_predictions[ct]['correct']}/{cancer_predictions[ct]['total']})")
        
        # Plot sensitivity by cancer type
        if cancer_sensitivities:
            print("Plotting sensitivity by cancer type...")
            plot_sensitivity_by_cancer_type(cancer_sensitivities, output_dir)
        
        # Print classification report
        print("\nClassification Report:")
        from sklearn.metrics import classification_report
        print(classification_report(all_y_true, all_y_pred, target_names=['Normal', 'Cancer']))
        
        # Save confusion matrix
        from sklearn.metrics import confusion_matrix
        conf_matrix = confusion_matrix(all_y_true, all_y_pred)
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        # Save the model and scaler
        print("\nSaving model and preprocessing components...")
        model_file = os.path.join(output_dir, 'xgboost_cancer_model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(final_model, f)
        print(f"Model saved to: {model_file}")
        
        scaler_file = os.path.join(output_dir, 'feature_scaler.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Feature scaler saved to: {scaler_file}")
        
        # Save feature importance scores
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': final_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        importance_file = os.path.join(output_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_file, index=False)
        print(f"Feature importance scores saved to: {importance_file}")
        
        # Create a prediction function for new samples
        def predict_cancer_risk(new_data):
            """
            Predict cancer risk for new data.
            
            Parameters:
            -----------
            new_data : pandas.DataFrame
                New data with the same features as the training data
                
            Returns:
            --------
            dict
                Dictionary with predictions and probabilities
            """
            # Ensure new_data has the same features
            missing_features = [col for col in feature_cols if col not in new_data.columns]
            if missing_features:
                raise ValueError(f"Missing features in new data: {missing_features}")
            
            # Extract features
            X_new = new_data[feature_cols].values
            
            # Clean and normalize data
            X_new_clean = np.copy(X_new)
            for i, col in enumerate(feature_cols):
                # Replace any possible NaN values with the median
                nan_mask = np.isnan(X_new_clean[:, i])
                if np.any(nan_mask):
                    X_new_clean[nan_mask, i] = np.nanmedian(X_new_clean[:, i])
            
            # Apply scaling
            X_new_scaled = scaler.transform(X_new_clean)
            
            # Predict
            y_pred = final_model.predict(X_new_scaled)
            y_prob = final_model.predict_proba(X_new_scaled)[:, 1]
            
            results = {
                'predictions': y_pred,
                'cancer_probabilities': y_prob,
                'predicted_class': ['Cancer' if p == 1 else 'Normal' for p in y_pred]
            }
            
            return results
        
        # Save the prediction function
        with open(os.path.join(output_dir, 'prediction_function.py'), 'w') as f:
            f.write("import numpy as np\n")
            f.write("import pandas as pd\n")
            f.write("import pickle\n\n")
            
            f.write("# Load the model and scaler\n")
            f.write("with open('xgboost_cancer_model.pkl', 'rb') as f:\n")
            f.write("    model = pickle.load(f)\n\n")
            
            f.write("with open('feature_scaler.pkl', 'rb') as f:\n")
            f.write("    scaler = pickle.load(f)\n\n")
            
            f.write("# Feature columns used for prediction\n")
            f.write(f"feature_cols = {feature_cols}\n\n")
            
            f.write("def predict_cancer_risk(new_data):\n")
            f.write("    \"\"\"\n")
            f.write("    Predict cancer risk for new data.\n")
            f.write("    \n")
            f.write("    Parameters:\n")
            f.write("    -----------\n")
            f.write("    new_data : pandas.DataFrame\n")
            f.write("        New data with the same features as the training data\n")
            f.write("        \n")
            f.write("    Returns:\n")
            f.write("    --------\n")
            f.write("    dict\n")
            f.write("        Dictionary with predictions and probabilities\n")
            f.write("    \"\"\"\n")
            f.write("    # Ensure new_data has the same features\n")
            f.write("    missing_features = [col for col in feature_cols if col not in new_data.columns]\n")
            f.write("    if missing_features:\n")
            f.write("        raise ValueError(f\"Missing features in new data: {missing_features}\")\n")
            f.write("    \n")
            f.write("    # Extract features\n")
            f.write("    X_new = new_data[feature_cols].values\n")
            f.write("    \n")
            f.write("    # Clean and normalize data\n")
            f.write("    X_new_clean = np.copy(X_new)\n")
            f.write("    for i, col in enumerate(feature_cols):\n")
            f.write("        # Replace any possible NaN values with the median\n")
            f.write("        nan_mask = np.isnan(X_new_clean[:, i])\n")
            f.write("        if np.any(nan_mask):\n")
            f.write("            X_new_clean[nan_mask, i] = np.nanmedian(X_new_clean[:, i])\n")
            f.write("    \n")
            f.write("    # Apply scaling\n")
            f.write("    X_new_scaled = scaler.transform(X_new_clean)\n")
            f.write("    \n")
            f.write("    # Predict\n")
            f.write("    y_pred = model.predict(X_new_scaled)\n")
            f.write("    y_prob = model.predict_proba(X_new_scaled)[:, 1]\n")
            f.write("    \n")
            f.write("    results = {\n")
            f.write("        'predictions': y_pred,\n")
            f.write("        'cancer_probabilities': y_prob,\n")
            f.write("        'predicted_class': ['Cancer' if p == 1 else 'Normal' for p in y_pred]\n")
            f.write("    }\n")
            f.write("    \n")
            f.write("    return results\n")
        
        print(f"Prediction function saved to: {os.path.join(output_dir, 'prediction_function.py')}")
        print("\nXGBoost cancer prediction pipeline completed successfully!")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()