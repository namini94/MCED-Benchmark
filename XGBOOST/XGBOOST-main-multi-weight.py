import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_class_weight
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

# Protein feature names - Keeping this part the same as your code
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

# Keeping your data processing functions the same
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
        if col not in ['PatientID', 'SampleID', 'Condition', 'Stage', 'Age']:
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

def plot_cancer_type_prediction_accuracy(class_names, top1_accuracies, top2_accuracies, 
                                         errors, output_dir='cancer_xgboost_multi', 
                                         filename='cancer_type_prediction.pdf'):
    """
    Plot cancer type prediction accuracy in a stacked bar chart similar to the provided figure.
    
    Parameters:
    -----------
    class_names : list
        Names of the cancer types
    top1_accuracies : list
        Accuracy of the top prediction for each cancer type
    top2_accuracies : list
        Accuracy of the top 2 predictions for each cancer type (cumulative)
    errors : list
        95% confidence intervals for the top2 accuracies
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
    
    # Calculate the top 2 only component (removing the top 1 part)
    top2_only = [top2 - top1 for top1, top2 in zip(top1_accuracies, top2_accuracies)]
    
    # Convert to percentages
    top1_pct = [acc * 100 for acc in top1_accuracies]
    top2_only_pct = [acc * 100 for acc in top2_only]
    errors_pct = [err * 100 for err in errors]
    
    # Create a dictionary mapping from class names to their data
    class_data = {}
    for i, class_name in enumerate(class_names):
        class_data[class_name] = {
            'top1': top1_pct[i],
            'top2_only': top2_only_pct[i],
            'error': errors_pct[i]
        }
    
    # Define the specific order requested
    ordered_classes = ["Colorectum", "Ovary", "Pancreas", "Breast", "Upper GI", "Lung", "Liver"]
    
    # Filter to only include classes that exist in our data
    ordered_classes = [cls for cls in ordered_classes if cls in class_data or cls.lower() in [c.lower() for c in class_data]]
    
    # Map class names if case doesn't match
    class_map = {}
    for c in class_data:
        for o in ordered_classes:
            if c.lower() == o.lower():
                class_map[o] = c
    ordered_classes = [class_map.get(c, c) for c in ordered_classes]
    
    # Add any missing classes at the end (though this shouldn't happen with the correct mapping)
    for cls in class_data:
        if cls not in ordered_classes:
            ordered_classes.append(cls)
    
    # Extract the ordered data
    ordered_top1 = [class_data[cls]['top1'] for cls in ordered_classes]
    ordered_top2_only = [class_data[cls]['top2_only'] for cls in ordered_classes]
    ordered_errors = [class_data[cls]['error'] for cls in ordered_classes]
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Define bar width
    bar_width = 0.7
    
    # Create positions for the bars
    indices = np.arange(len(ordered_classes))
    
    # Plot the stacked bars
    bars1 = plt.bar(indices, ordered_top1, bar_width, color='lightblue', label='Top Prediction')
    bars2 = plt.bar(indices, ordered_top2_only, bar_width, bottom=ordered_top1, color='steelblue', label='Top 2 Predictions')
    
    # Add value labels on the bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # Label for top1
        plt.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height()/2.,
                 f'{ordered_top1[i]:.0f}%', ha='center', va='center',
                 color='black', fontweight='bold')
        
        # Label for combined top1+top2
        total_height = bar1.get_height() + bar2.get_height()
        plt.text(bar2.get_x() + bar2.get_width()/2., bar1.get_height() + bar2.get_height()/2.,
                 f'{total_height:.0f}%', ha='center', va='center',
                 color='white', fontweight='bold')
    
    # Add error bars to the top of the stacked bars
    plt.errorbar(indices, [sum(x) for x in zip(ordered_top1, ordered_top2_only)], 
                 yerr=ordered_errors, fmt='none', ecolor='black', 
                 capsize=5, capthick=2, elinewidth=2)
    
    # Set chart properties
    plt.xlabel('Cancer Type', fontsize=14)
    plt.ylabel('Accuracy of predictor (%)', fontsize=14)
    plt.title('Identification of cancer type by XGBOOST (with class weighting)', 
              fontsize=16, fontweight='bold')
    plt.xticks(indices, ordered_classes, rotation=0)
    plt.yticks(np.arange(0, 101, 20))
    plt.ylim(0, 105)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add a tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


# New function to calculate class weights
def calculate_class_weights(y_labels, class_names):
    """
    Calculate class weights inversely proportional to class frequencies.
    
    Parameters:
    -----------
    y_labels : array-like
        Target labels
    class_names : array-like
        Names of the classes
        
    Returns:
    --------
    dict
        Dictionary with class indices as keys and weights as values
    """
    # Calculate class weights using sklearn's built-in function
    classes = np.unique(y_labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_labels)
    
    # Create a dictionary of class weights
    weight_dict = {i: weight for i, weight in zip(classes, class_weights)}
    
    print("\nClass weights to handle imbalance:")
    for class_idx, weight in weight_dict.items():
        print(f"  {class_names[class_idx]}: {weight:.4f}")
    
    return weight_dict

def main():
    """Main function to run the XGBoost cancer classification pipeline."""
    try:
        # Create output directory
        output_dir = 'cancer_xgboost_multi_weighted'
        os.makedirs(output_dir, exist_ok=True)
        
        # Read data
        print("Reading data...")
        try:
            data = pd.read_excel("PositiveCancerSEEKResults.xlsx")
        except FileNotFoundError:
            print("PositiveCancerSEEKResults.xlsx not found. Please make sure the file exists in the current directory.")
            return
        
        # Print basic dataset information
        print(f"Dataset shape: {data.shape}")
        
        # Check if 'Condition' column exists
        if 'Condition' not in data.columns:
            print("Error: 'Condition' column not found in the dataset.")
            return
            
        print(f"Cancer types present: {data['Condition'].unique()}")
        print(f"Number of cancer types: {data['Condition'].nunique()}")
        
        # Print class distribution
        print("\nClass distribution:")
        class_counts = data['Condition'].value_counts()
        for cancer_type, count in class_counts.items():
            print(f"  {cancer_type}: {count} samples ({count/len(data)*100:.1f}%)")
        
        # Clean data by removing stars (*) from numeric fields
        print("\nCleaning data (removing * characters)...")
        data_cleaned = clean_numeric_data(data)
        
        # Identify protein features in the dataset
        available_proteins = [protein for protein in PROTEIN_FEATURES if protein in data_cleaned.columns]
        print(f"Found {len(available_proteins)} protein features out of {len(PROTEIN_FEATURES)} expected")
        
        # Find the Omega score column and age column
        omega_cols = [col for col in data_cleaned.columns if 'Omega' in col]
        age_col = 'Age' if 'Age' in data_cleaned.columns else None
        
        if omega_cols:
            print(f"Found mutation feature: {omega_cols}")
        else:
            print("No Omega score column found. Model will only use protein features.")
            omega_cols = []
            
        if age_col:
            print(f"Found age feature: {age_col}")
        else:
            print("No Age column found.")
            age_col = None
        
        # All feature columns
        feature_cols = available_proteins + omega_cols
        if age_col:
            feature_cols.append(age_col)
        
        # Apply log2 normalization
        print("Applying log2 normalization...")
        data_log2 = log2_normalize(data_cleaned)
        
        # Remove rows with NaN values in feature columns
        print("Removing rows with NaN values...")
        data_clean = remove_nan_rows(data_log2, feature_cols)
        
        # Extract features and labels from cleaned data
        X = data_clean[feature_cols].values
        
        # Encode cancer types
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data_clean['Condition'])
        class_names = label_encoder.classes_
        n_classes = len(class_names)
        
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of samples after cleaning: {X.shape[0]}")
        print(f"Number of classes: {n_classes}")
        print(f"Class names: {class_names}")
        
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
        
        # Save the preprocessed data for future use
        print("\nSaving preprocessed data...")
        preprocessed_data = pd.DataFrame(X_scaled, columns=feature_cols)
        preprocessed_data['Condition'] = data_clean['Condition'].values  # Add cancer type labels
        if 'PatientID' in data_clean.columns:
            preprocessed_data['PatientID'] = data_clean['PatientID'].values  # Add patient IDs for reference
        
        # Save to CSV
        preprocessed_file = os.path.join(output_dir, 'preprocessed_data.csv')
        preprocessed_data.to_csv(preprocessed_file, index=False)
        print(f"Preprocessed data saved to: {preprocessed_file}")
        
        # Cross-validation setup
        n_splits = 5
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        
        # Store results
        fold_accuracies = []
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        # For tracking class-wise performance
        class_top1_correct = {i: 0 for i in range(n_classes)}
        class_top2_correct = {i: 0 for i in range(n_classes)}
        class_total = {i: 0 for i in range(n_classes)}
        
        # Calculate overall class weights based on the entire dataset
        overall_class_weights = calculate_class_weights(y, class_names)
        
        # XGBoost parameters with regularization to prevent overfitting
        params = {
            'objective': 'multi:softprob',  # multiclass classification with probability output
            'num_class': n_classes,
            'learning_rate': 0.05,
            'max_depth': 4,
            'min_child_weight': 2,
            'gamma': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,
            'reg_lambda': 10.0,
            'n_estimators': 100,
            'random_state': RANDOM_SEED,
            'use_label_encoder': False,
            'eval_metric': 'mlogloss'
        }
        
        # Perform cross-validation
        print(f"\nPerforming {n_splits}-fold cross-validation with class weighting...")
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled, y)):
            print(f"\nFold {fold+1}/{n_splits}")
            
            # Split data
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Print class distribution in training set
            unique, counts = np.unique(y_train, return_counts=True)
            print("Class distribution in training set:")
            for class_idx, count in zip(unique, counts):
                print(f"  {class_names[class_idx]}: {count} samples")
            
            # Calculate class weights for this fold
            fold_class_weights = calculate_class_weights(y_train, class_names)
            
            # Create and train XGBoost model
            print(f"Training XGBoost model for fold {fold+1} with class weights...")
            
            # Create sample weights array based on class weights
            sample_weights = np.array([fold_class_weights[label] for label in y_train])
            
            model = xgb.XGBClassifier(**params)
            
            # Train the model with sample weights
            model.fit(
                X_train, 
                y_train,
                sample_weight=sample_weights,  # Add sample weights for class imbalance
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Predict on test set
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Store predictions for later analysis
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(y_proba)
            
            # Track top-1 and top-2 predictions for each class
            for i, true_label in enumerate(y_test):
                # Increment total count for this class
                class_total[true_label] += 1
                
                # Check if top-1 prediction is correct
                if y_pred[i] == true_label:
                    class_top1_correct[true_label] += 1
                
                # Get top-2 predictions
                top2_indices = np.argsort(y_proba[i])[-2:]
                if true_label in top2_indices:
                    class_top2_correct[true_label] += 1
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            fold_accuracies.append(accuracy)
            
            print(f"Fold {fold+1} accuracy: {accuracy:.4f}")
        
        # Calculate overall accuracy
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        print(f"\nAverage accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        
        # Calculate top-1 and top-2 accuracies for each class
        top1_accuracies = []
        top2_accuracies = []
        errors = []  # For 95% confidence intervals
        
        print("\nPer-class performance:")
        for i in range(n_classes):
            if class_total[i] > 0:
                top1_acc = class_top1_correct[i] / class_total[i]
                top2_acc = class_top2_correct[i] / class_total[i]
                
                # Calculate 95% confidence interval
                error = 1.96 * np.sqrt((top2_acc * (1 - top2_acc)) / class_total[i])
            else:
                top1_acc = 0
                top2_acc = 0
                error = 0
            
            top1_accuracies.append(top1_acc)
            top2_accuracies.append(top2_acc)
            errors.append(error)
            
            print(f"{class_names[i]}:")
            print(f"  Total samples: {class_total[i]}")
            print(f"  Top-1 accuracy: {top1_acc:.4f} ({class_top1_correct[i]}/{class_total[i]})")
            print(f"  Top-2 accuracy: {top2_acc:.4f} ({class_top2_correct[i]}/{class_total[i]})")
            print(f"  95% CI: ±{error:.4f}")
        
        # Plot the cancer type prediction accuracy chart
        print("\nPlotting cancer type prediction accuracy chart...")
        plot_cancer_type_prediction_accuracy(
            class_names=class_names,
            top1_accuracies=top1_accuracies,
            top2_accuracies=top2_accuracies,
            errors=errors,
            output_dir=output_dir
        )
        
        # Print overall classification report
        print("\nClassification Report:")
        print(classification_report(all_y_true, all_y_pred, target_names=class_names))
        
        # Create confusion matrix
        cm = confusion_matrix(all_y_true, all_y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Save confusion matrix to CSV for easier viewing
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_file = os.path.join(output_dir, 'confusion_matrix.csv')
        cm_df.to_csv(cm_file)
        print(f"Confusion matrix saved to: {cm_file}")
        
        # Train a final model on all data with class weights
        print("\nTraining final model on all data with class weights...")
        
        # Apply weights to entire dataset
        sample_weights_all = np.array([overall_class_weights[label] for label in y])
        
        final_model = xgb.XGBClassifier(**params)
        final_model.fit(X_scaled, y, sample_weight=sample_weights_all)
        
        # Save the final model and label encoder
        model_file = os.path.join(output_dir, 'xgboost_cancer_type_weighted_model.pkl')
        with open(model_file, 'wb') as f:
            pickle.dump(final_model, f)
        
        label_encoder_file = os.path.join(output_dir, 'label_encoder.pkl')
        with open(label_encoder_file, 'wb') as f:
            pickle.dump(label_encoder, f)
        
        scaler_file = os.path.join(output_dir, 'feature_scaler.pkl')
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save class weights for future reference
        weights_file = os.path.join(output_dir, 'class_weights.pkl')
        with open(weights_file, 'wb') as f:
            pickle.dump(overall_class_weights, f)
        
        print(f"Model, label encoder, scaler, and class weights saved to: {output_dir}")
        
        # Create prediction function
        with open(os.path.join(output_dir, 'predict_cancer_type.py'), 'w') as f:
            f.write("import pickle\n")
            f.write("import numpy as np\n")
            f.write("import pandas as pd\n\n")
            
            f.write("# Load model, label encoder, and scaler\n")
            f.write("with open('xgboost_cancer_type_weighted_model.pkl', 'rb') as f:\n")
            f.write("    model = pickle.load(f)\n\n")
            
            f.write("with open('label_encoder.pkl', 'rb') as f:\n")
            f.write("    label_encoder = pickle.load(f)\n\n")
            
            f.write("with open('feature_scaler.pkl', 'rb') as f:\n")
            f.write("    scaler = pickle.load(f)\n\n")
            
            f.write("with open('class_weights.pkl', 'rb') as f:\n")
            f.write("    class_weights = pickle.load(f)\n\n")
            
            f.write(f"# Feature list\nfeature_cols = {feature_cols}\n\n")
            
            f.write("def predict_cancer_type(sample_data, top_k=2):\n")
            f.write("    \"\"\"\n")
            f.write("    Predict cancer type for new sample data.\n")
            f.write("    \n")
            f.write("    Parameters:\n")
            f.write("    -----------\n")
            f.write("    sample_data : pandas.DataFrame\n")
            f.write("        Sample data with features matching the training data\n")
            f.write("    top_k : int, optional (default=2)\n")
            f.write("        Number of top predictions to return\n")
            f.write("        \n")
            f.write("    Returns:\n")
            f.write("    --------\n")
            f.write("    list\n")
            f.write("        List of dictionaries with top predictions and probabilities\n")
            f.write("    \"\"\"\n")
            f.write("    # Check that all required features are present\n")
            f.write("    missing_features = [f for f in feature_cols if f not in sample_data.columns]\n")
            f.write("    if missing_features:\n")
            f.write("        raise ValueError(f\"Missing features: {missing_features}\")\n\n")
            
            f.write("    # Extract features\n")
            f.write("    X = sample_data[feature_cols].values\n\n")
            
            f.write("    # Handle NaN values\n")
            f.write("    X = np.nan_to_num(X, nan=np.nanmedian(X))\n\n")
            
            f.write("    # Apply scaling\n")
            f.write("    X_scaled = scaler.transform(X)\n\n")
            
            f.write("    # Predict probabilities\n")
            f.write("    proba = model.predict_proba(X_scaled)\n\n")
            
            f.write("    # Get top-k predictions for each sample\n")
            f.write("    results = []\n")
            f.write("    for i, p in enumerate(proba):\n")
            f.write("        top_indices = np.argsort(p)[-top_k:][::-1]\n")
            f.write("        top_probs = p[top_indices]\n")
            f.write("        top_labels = [label_encoder.classes_[idx] for idx in top_indices]\n")
            f.write("        \n")
            f.write("        results.append({\n")
            f.write("            'sample_idx': i,\n")
            f.write("            'top_labels': top_labels,\n")
            f.write("            'top_probabilities': top_probs\n")
            f.write("        })\n\n")
            
            f.write("    return results\n")
        
        print(f"Prediction function saved to: {os.path.join(output_dir, 'predict_cancer_type.py')}")
        print("\nMulticlass cancer type prediction pipeline with class weighting completed successfully!")
    
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()