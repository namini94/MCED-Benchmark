import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import umap.umap_ as umap
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

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

PROTEIN_FEATURES = [
    'AFP (pg/ml)', 'Angiopoietin-2 (pg/ml)', 'AXL (pg/ml)', 'CA-125 (U/ml)',
    'CA 15-3 (U/ml)', 'CA19-9 (U/ml)', 'CD44 (ng/ml)', 'CEA (pg/ml)',
    'CYFRA 21-1 (pg/ml)', 'DKK1 (ng/ml)', 'Endoglin (pg/ml)', 'FGF2 (pg/ml)',
    'Follistatin (pg/ml)', 'Galectin-3 (ng/ml)', 'G-CSF (pg/ml)', 'GDF15 (ng/ml)',
    'HE4 (pg/ml)', 'HGF (pg/ml)', 'IL-6 (pg/ml)', 'IL-8 (pg/ml)',
    'Kallikrein-6 (pg/ml)', 'Leptin (pg/ml)', 'Mesothelin (ng/ml)', 'Midkine (pg/ml)',
    'Myeloperoxidase (ng/ml)', 'NSE (ng/ml)', 'OPG (ng/ml)', 'OPN (pg/ml)',
    'PAR (pg/ml)', 'Prolactin (pg/ml)', 'sEGFR (pg/ml)', 'sFas (pg/ml)',
    'SHBG (nM)', 'sHER2/sEGFR2/sErbB2 (pg/ml)', 'sPECAM-1 (pg/ml)', 'TGFa (pg/ml)',
    'Thrombospondin-2 (pg/ml)', 'TIMP-1 (pg/ml)', 'TIMP-2 (pg/ml)'
]

def clean_numeric_data(data):
    cleaned_data = data.copy()
    for col in cleaned_data.columns:
        if col not in ['PatientID', 'SampleID', 'Condition', 'Stage']:
            if cleaned_data[col].dtype == object or isinstance(cleaned_data[col].iloc[0], str):
                cleaned_data[col] = cleaned_data[col].astype(str).str.replace('*', '')
                cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                nan_count = cleaned_data[col].isna().sum()
                if nan_count > 0:
                    print(f"Warning: {nan_count} values in column '{col}' could not be converted to numeric")
            elif pd.api.types.is_numeric_dtype(cleaned_data[col]):
                if cleaned_data[col].astype(str).str.contains('\*').any():
                    cleaned_data[col] = cleaned_data[col].astype(str).str.replace('*', '')
                    cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
    return cleaned_data

def log2_normalize(data):
    data_norm = data.copy()
    for col in data_norm.columns:
        if col not in ['PatientID', 'SampleID', 'Condition', 'Stage', 'Age']:
            if pd.api.types.is_numeric_dtype(data_norm[col]):
                min_val = data_norm[col].min()
                if pd.isna(min_val):
                    print(f"Warning: Column '{col}' contains NaN values, filling with column median")
                    median_val = data_norm[col].median()
                    data_norm[col] = data_norm[col].fillna(median_val)
                    min_val = data_norm[col].min()
                if min_val <= 0:
                    offset = abs(min_val) + 1
                    data_norm[col] = np.log2(data_norm[col] + offset)
                else:
                    data_norm[col] = np.log2(data_norm[col])
    return data_norm

def remove_nan_rows(data, feature_cols):
    original_count = len(data)
    clean_data = data.dropna(subset=feature_cols)
    removed_count = original_count - len(clean_data)
    if removed_count > 0:
        print(f"Removed {removed_count} rows with NaN values ({removed_count/original_count*100:.2f}% of data)")
        print(f"Final dataset size: {len(clean_data)} samples")
    return clean_data

class DualDecoderVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10, hidden_dims=[128, 64], num_classes=2):
        super(DualDecoderVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.LeakyReLU(0.2))
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Mean and variance for latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder 1 - Reconstruction
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.LeakyReLU(0.2))
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Decoder 2 - Classification
        classifier_layers = []
        prev_dim = latent_dim
        for i, hidden_dim in enumerate(reversed(hidden_dims)):
            if i == 0:
                classifier_layers.append(nn.Linear(prev_dim, hidden_dim // 2))
                classifier_layers.append(nn.BatchNorm1d(hidden_dim // 2))
                classifier_layers.append(nn.LeakyReLU(0.2))
                prev_dim = hidden_dim // 2
            else:
                classifier_layers.append(nn.Linear(prev_dim, hidden_dim // 2))
                classifier_layers.append(nn.BatchNorm1d(hidden_dim // 2))
                classifier_layers.append(nn.LeakyReLU(0.2))
                prev_dim = hidden_dim // 2
        
        classifier_layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
    
    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        return self.decoder(z)
    
    def classify(self, z):
        return self.classifier(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        class_logits = self.classify(z)
        return x_recon, class_logits, mu, log_var
    
    def get_embeddings(self, x):
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu

def vae_classification_loss(x_recon, x, class_logits, labels, mu, log_var, beta=1.0, lambda_cls=1.0):
    # Reconstruction loss
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Classification loss
    cls_loss = F.cross_entropy(class_logits, labels, reduction='sum')
    
    # Total loss: ELBO + classification
    total_loss = recon_loss + beta * kl_loss + lambda_cls * cls_loss
    
    return total_loss, recon_loss, kl_loss, cls_loss

def train_dual_vae(model, dataloader, optimizer, device, epochs=100, beta=1.0, lambda_cls=1.0, scheduler=None):
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        recon_loss_sum = 0
        kl_loss_sum = 0
        cls_loss_sum = 0
        
        for batch_idx, data in enumerate(dataloader):
            x, y = data[0].to(device), data[1].to(device)
            
            # Forward pass
            x_recon, class_logits, mu, log_var = model(x)
            
            # Calculate loss
            loss, recon_loss, kl_loss, cls_loss = vae_classification_loss(
                x_recon, x, class_logits, y, mu, log_var, beta, lambda_cls
            )
            
            # Skip backpropagation if loss is NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered in batch {batch_idx}. Skipping update.")
                continue
                
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            batch_loss = loss.item()
            if not np.isnan(batch_loss):
                epoch_loss += batch_loss
                recon_loss_sum += recon_loss.item()
                kl_loss_sum += kl_loss.item()
                cls_loss_sum += cls_loss.item()
        
        # Calculate average loss for the epoch
        n_samples = len(dataloader.dataset)
        avg_loss = epoch_loss / n_samples if n_samples > 0 else float('nan')
        avg_recon_loss = recon_loss_sum / n_samples if n_samples > 0 else float('nan')
        avg_kl_loss = kl_loss_sum / n_samples if n_samples > 0 else float('nan')
        avg_cls_loss = cls_loss_sum / n_samples if n_samples > 0 else float('nan')
        
        # Store losses
        losses.append((avg_loss, avg_recon_loss, avg_kl_loss, avg_cls_loss))
        
        # Step the scheduler if provided
        if scheduler is not None and not np.isnan(avg_loss):
            scheduler.step(avg_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Recon: {avg_recon_loss:.4f}, "
                  f"KL: {avg_kl_loss:.4f}, Cls: {avg_cls_loss:.4f}")
    
    return losses

def get_vae_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            x = data[0].to(device)
            mu, _ = model.encode(x)
            if torch.isnan(mu).any():
                print(f"Warning: NaN values detected in embeddings for batch {batch_idx}")
                mu = torch.nan_to_num(mu, nan=0.0)
            embeddings.append(mu.cpu().numpy())
    
    return np.vstack(embeddings)

def get_classifier_predictions(model, dataloader, device):
    model.eval()
    predictions = []
    scores = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            x = data[0].to(device)
            _, class_logits, _, _ = model(x)
            probs = F.softmax(class_logits, dim=1)
            preds = torch.argmax(class_logits, dim=1)
            predictions.append(preds.cpu().numpy())
            scores.append(probs.cpu().numpy())
    
    return np.concatenate(predictions), np.concatenate(scores)

def plot_training_losses(losses, output_dir='dual_vae_multi_results', filename='training_losses.pdf'):
    """
    Plot training losses for Dual VAE.
    
    Parameters:
    -----------
    losses : list
        List of loss values (total, reconstruction, KL, classification)
    output_dir : str
        Directory to save visualizations
    filename : str
        Filename for the visualization
        
    Returns:
    --------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    total_losses = [l[0] for l in losses]
    recon_losses = [l[1] for l in losses]
    kl_losses = [l[2] for l in losses]
    cls_losses = [l[3] for l in losses]
    
    if any(np.isnan(l) for l in total_losses) or any(np.isnan(l) for l in recon_losses) or \
       any(np.isnan(l) for l in kl_losses) or any(np.isnan(l) for l in cls_losses):
        print("Warning: NaN values found in losses. Plot may be incomplete.")
    
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(losses) + 1)
    plt.plot(epochs, total_losses, 'b-', label='Total Loss', linewidth=2)
    plt.plot(epochs, recon_losses, 'g-', label='Reconstruction Loss', linewidth=2)
    plt.plot(epochs, kl_losses, 'r-', label='KL Divergence', linewidth=2)
    plt.plot(epochs, cls_losses, 'y-', label='Classification Loss', linewidth=2)
    
    plt.title('Dual Decoder VAE Training Losses', fontsize=18, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fold_accuracies(fold_accuracies, output_dir='dual_vae_multi_results', filename='fold_accuracies.pdf'):
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
    plt.title('Cross-Validation Accuracy by Fold', fontsize=18, fontweight='bold')
    plt.xticks(fold_nums)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def plot_cancer_type_prediction_accuracy(class_names, top1_accuracies, top2_accuracies, 
                                         errors, output_dir='dual_vae_multi_results', 
                                         filename='cancer_type_prediction.pdf'):
    """
    Plot cancer type prediction accuracy similar to the XGBoost version.
    
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
    plt.title('Identification of cancer type by Dual VAE', 
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

def plot_latent_space_by_cancer_type(embeddings, labels, class_names, output_dir='dual_vae_multi_results', filename='latent_space_cancer_types.pdf'):
    """
    Plot UMAP visualization of the latent space colored by cancer type.
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        Latent space embeddings
    labels : numpy.ndarray
        Class labels
    class_names : list
        Names of the cancer types
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
    
    if np.isnan(embeddings).any():
        print("Warning: NaN values found in embeddings. Replacing with zeros.")
        embeddings = np.nan_to_num(embeddings)
    
    # Apply UMAP for dimensionality reduction
    reducer = umap.UMAP(random_state=RANDOM_SEED)
    embedding_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(14, 12))
    
    # Set up a colormap
    num_classes = len(class_names)
    cmap = plt.cm.get_cmap('tab20', num_classes)
    
    # Plot each class
    for i, class_name in enumerate(class_names):
        idx = labels == i
        plt.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1], 
                    c=[cmap(i)], label=class_name, alpha=0.7, edgecolors='none', s=80)
    
    plt.title('UMAP Visualization of Dual VAE Latent Space by Cancer Type', fontsize=18, fontweight='bold')
    plt.xlabel('UMAP 1', fontsize=14)
    plt.ylabel('UMAP 2', fontsize=14)
    
    # Add legend with two columns for better layout
    plt.legend(title='Cancer Type', title_fontsize=14, fontsize=12, 
               markerscale=1.5, loc='best', ncol=2)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_feature_importance_shap(model, X_data, feature_names, class_names, output_dir='dual_vae_multi_results', n_background=100):
    """
    Analyze feature importance using SHAP values for multi-class model.
    
    Parameters:
    -----------
    model : DualDecoderVAE
        Trained Dual Decoder VAE model
    X_data : numpy.ndarray
        Data for explanation
    feature_names : list
        List of feature names
    class_names : list
        Names of the cancer types
    output_dir : str
        Directory to save visualizations
    n_background : int
        Number of background samples for SHAP explainer
        
    Returns:
    --------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert model to eval mode
    model.eval()
    
    # Create a function that takes in raw data and returns class probabilities
    def f(x):
        x_tensor = torch.FloatTensor(x).to(next(model.parameters()).device)
        with torch.no_grad():
            _, class_logits, _, _ = model(x_tensor)
            probas = torch.softmax(class_logits, dim=1).cpu().numpy()
        return probas  # Return probabilities for all classes
    
    # Choose a subset of background examples for the explainer
    background_indices = np.random.choice(X_data.shape[0], min(n_background, X_data.shape[0]), replace=False)
    background_data = X_data[background_indices]
    
    # Create KernelExplainer
    print("Creating SHAP explainer...")
    explainer = shap.KernelExplainer(f, background_data)
    
    # Choose a subset of samples to analyze for reasonable computation time
    max_samples = min(200, X_data.shape[0])
    indices = np.random.choice(X_data.shape[0], max_samples, replace=False)
    samples = X_data[indices]
    
    # Calculate SHAP values
    print(f"Calculating SHAP values for {max_samples} samples...")
    shap_values = explainer.shap_values(samples)
    
    # Calculate mean absolute SHAP values for ranking features (across all classes)
    mean_abs_shap_all_classes = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    mean_abs_shap = np.mean(mean_abs_shap_all_classes, axis=0)
    
    feature_importance = pd.DataFrame(list(zip(feature_names, mean_abs_shap)), 
                                      columns=['Feature','SHAP Importance'])
    feature_importance.sort_values(by=['SHAP Importance'], ascending=False, inplace=True)
    
    # Plot feature importance as a bar chart
    plt.figure(figsize=(12, 10))
    plt.barh(feature_importance['Feature'][:20], feature_importance['SHAP Importance'][:20])
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Top 20 Features Ranked by SHAP Importance (All Classes)')
    plt.gca().invert_yaxis()  # Invert y-axis to have highest importance at the top
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_feature_importance_bar.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Use SHAP's summary plot with bar type
    plt.figure(figsize=(14, 10))
    # Use the aggregate SHAP values across all classes for the summary
    shap.summary_plot(np.array(shap_values), samples, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (All Classes)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_feature_importance.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot SHAP summary plot (dot plot) for top 20 features
    plt.figure(figsize=(14, 12))
    shap.summary_plot(np.array(shap_values), samples, feature_names=feature_names, max_display=20, show=False)
    plt.title("SHAP Values for Top 20 Features (All Classes)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_top_features.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save top 20 features to a text file
    with open(os.path.join(output_dir, 'top_features_by_shap.txt'), 'w') as f:
        f.write("Rank\tFeature\tSHAP Importance\n")
        for i, (feature, importance) in enumerate(zip(feature_importance['Feature'][:20], 
                                                     feature_importance['SHAP Importance'][:20])):
            f.write(f"{i+1}\t{feature}\t{importance:.6f}\n")
    
        # Create class-specific visualizations for top classes
    for i, class_name in enumerate(class_names):
        if i >= 5:  # Limit to first 5 classes for brevity
            break
            
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values[i], samples, feature_names=feature_names, 
                         max_display=10, show=False, plot_type="bar")
        plt.title(f"SHAP Feature Importance for {class_name}", fontsize=16)
        plt.tight_layout()
        safe_name = class_name.replace(" ", "_").replace("/", "_").replace("-", "_")
        plt.savefig(os.path.join(output_dir, f'shap_importance_{safe_name}.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
def main():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        output_dir = 'dual_vae_multi_results'
        os.makedirs(output_dir, exist_ok=True)
        
        print("Reading data...")
        try:
            data = pd.read_excel("PositiveCancerSEEKResults.xlsx")
        except FileNotFoundError:
            print("PositiveCancerSEEKResults.xlsx not found. Trying Dataset.xlsx...")
            try:
                data = pd.read_excel("Dataset.xlsx")
                # Filter to keep only cancer samples (exclude Normal)
                data = data[data['Condition'] != 'Normal']
            except FileNotFoundError:
                print("Neither PositiveCancerSEEKResults.xlsx nor Dataset.xlsx found. Please ensure one exists.")
                return
        
        if 'Condition' not in data.columns:
            raise ValueError("Dataset must contain a 'Condition' column with cancer types")
        
        print(f"Dataset shape: {data.shape}")
        print(f"Cancer types: {data['Condition'].unique()}")
        print(f"Number of cancer types: {data['Condition'].nunique()}")
        
        # Print class distribution
        print("\nClass distribution:")
        class_counts = data['Condition'].value_counts()
        for cancer_type, count in class_counts.items():
            print(f"  {cancer_type}: {count} samples ({count/len(data)*100:.1f}%)")
        
        print("\nCleaning data (removing * characters)...")
        data_cleaned = clean_numeric_data(data)
        
        available_proteins = [protein for protein in PROTEIN_FEATURES if protein in data_cleaned.columns]
        
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
        
        print(f"Using {len(feature_cols)} features: {feature_cols}")
        
        print("Applying log2 normalization...")
        data_log2 = log2_normalize(data_cleaned)
        
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
        
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values still present in data after cleaning.")
            print("Imputing remaining NaN values with feature medians...")
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
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
        
        # Hyperparameters
        latent_dim = 8
        hidden_dims = [64, 32]
        
        # Cross-validation setup
        n_splits = 10
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        
        # Store results
        all_embeddings = []
        all_labels = []
        fold_accuracies = []
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        
        # For tracking class-wise performance
        class_top1_correct = {i: 0 for i in range(n_classes)}
        class_top2_correct = {i: 0 for i in range(n_classes)}
        class_total = {i: 0 for i in range(n_classes)}
        
        # Store models and test loaders
        models = []
        test_loaders = []
        fold_indices = []
        
        print(f"\nPerforming {n_splits}-fold cross-validation...")
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled, y)):
            print(f"\nFold {fold+1}/{n_splits}")
            
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Print class distribution in training set
            unique, counts = np.unique(y_train, return_counts=True)
            print("Class distribution in training set:")
            for class_idx, count in zip(unique, counts):
                print(f"  {class_names[class_idx]}: {count} samples")
            
            X_train_tensor = torch.FloatTensor(X_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_train_tensor = torch.LongTensor(y_train)
            y_test_tensor = torch.LongTensor(y_test)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            input_dim = X_train.shape[1]
            model = DualDecoderVAE(input_dim=input_dim, latent_dim=latent_dim, 
                                  hidden_dims=hidden_dims, num_classes=n_classes).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
            
            print(f"Training Dual Decoder VAE for fold {fold+1}...")
            losses = train_dual_vae(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                epochs=100,
                beta=5.0,
                lambda_cls=200.0,
                scheduler=scheduler
            )
            
            plot_training_losses(
                losses=losses,
                output_dir=output_dir,
                filename=f'training_losses_fold_{fold+1}.pdf'
            )
            
            # Get predictions from the built-in classifier
            y_pred, y_scores = get_classifier_predictions(model, test_loader, device)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            fold_accuracies.append(accuracy)
            
            # Get embeddings for visualization
            test_embeddings = get_vae_embeddings(model, test_loader, device)
            
            # Store embeddings and labels for later visualization
            all_embeddings.append(test_embeddings)
            all_labels.append(y_test)
            
            # Store all predictions for later analysis
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_proba.extend(y_scores)
            
            # Track top-1 and top-2 predictions for each class
            for i, true_label in enumerate(y_test):
                # Increment total count for this class
                class_total[true_label] += 1
                
                # Check if top-1 prediction is correct
                if y_pred[i] == true_label:
                    class_top1_correct[true_label] += 1
                
                # Get top-2 predictions
                top2_indices = np.argsort(y_scores[i])[-2:]
                if true_label in top2_indices:
                    class_top2_correct[true_label] += 1
            
            # Store model, test loader, and indices for later use
            models.append(model)
            test_loaders.append(test_loader)
            fold_indices.append((train_idx, test_idx))
            
            print(f"Fold {fold+1} accuracy: {accuracy:.4f}")
        
        # Calculate average metrics across folds
        avg_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)
        
        # Plot fold accuracies
        plot_fold_accuracies(fold_accuracies, output_dir)
        
        print(f"\nAverage accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        
        # Combine embeddings and labels from all folds
        combined_embeddings = np.vstack(all_embeddings)
        combined_labels = np.concatenate(all_labels)
        
        # Convert all_y_proba to numpy array for easier analysis
        all_y_proba = np.array(all_y_proba)
        
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
        
        # Plot cancer type prediction accuracy chart
        print("\nPlotting cancer type prediction accuracy chart...")
        plot_cancer_type_prediction_accuracy(
            class_names=class_names,
            top1_accuracies=top1_accuracies,
            top2_accuracies=top2_accuracies,
            errors=errors,
            output_dir=output_dir
        )
        
        # Visualize latent space by cancer type
        print("\nVisualizing latent space by cancer type...")
        plot_latent_space_by_cancer_type(combined_embeddings, combined_labels, class_names, output_dir)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_y_true, all_y_pred, target_names=class_names))
        
        # Create confusion matrix
        cm = confusion_matrix(all_y_true, all_y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot confusion matrix as heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save confusion matrix to CSV for easier viewing
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_file = os.path.join(output_dir, 'confusion_matrix.csv')
        cm_df.to_csv(cm_file)
        print(f"Confusion matrix saved to: {cm_file}")
        
        # Use the best fold model for SHAP analysis
        best_fold_idx = np.argmax([fold_accuracies[i] for i in range(len(fold_accuracies))])
        print(f"\nPerforming SHAP analysis using model from best fold (Fold {best_fold_idx+1})...")
        
        best_model = models[best_fold_idx]
        
        # Analyze feature importance using SHAP
        analyze_feature_importance_shap(
            model=best_model,
            X_data=X_scaled,
            feature_names=feature_cols,
            class_names=class_names,
            output_dir=output_dir,
            n_background=100
        )
        
        # Train a final model on all data
        print("\nTraining final model on all data...")
        final_model = DualDecoderVAE(
            input_dim=X_scaled.shape[1], 
            latent_dim=latent_dim, 
            hidden_dims=hidden_dims, 
            num_classes=n_classes
        ).to(device)
        
        all_tensor_X = torch.FloatTensor(X_scaled).to(device)
        all_tensor_y = torch.LongTensor(y).to(device)
        all_dataset = TensorDataset(all_tensor_X, all_tensor_y)
        all_loader = DataLoader(all_dataset, batch_size=32, shuffle=True)
        
        optimizer = optim.Adam(final_model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        train_dual_vae(
            model=final_model,
            dataloader=all_loader,
            optimizer=optimizer,
            device=device,
            epochs=100,
            beta=5.0,
            lambda_cls=20.0
        )
        
        # Save the model
        model_path = os.path.join(output_dir, 'dualvae_multiclass_model.pt')
        torch.save(final_model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        
        # Save scaler for preprocessing new data
        scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save label encoder
        encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"Label encoder saved to: {encoder_path}")
        
        # Save feature list
        with open(os.path.join(output_dir, 'feature_list.txt'), 'w') as f:
            for feature in feature_cols:
                f.write(f"{feature}\n")
        
        # Save results to a text file
        with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
            f.write("Dual Decoder VAE with Integrated Classification for Cancer Type Prediction\n")
            f.write("=======================================================================\n\n")
            
            f.write(f"Original dataset shape: {data.shape}\n")
            f.write(f"Final dataset shape after cleaning: {X.shape}\n")
            f.write(f"Number of features: {X.shape[1]}\n")
            f.write(f"Number of cancer types: {n_classes}\n\n")
            
            f.write("Cancer type distribution:\n")
            for cancer_type, count in class_counts.items():
                f.write(f"  {cancer_type}: {count} samples ({count/len(data)*100:.1f}%)\n")
            
            f.write(f"\nVAE latent dimension: {latent_dim}\n")
            f.write(f"KL divergence weight (beta): 5.0\n")
            f.write(f"Classification loss weight (lambda): 20.0\n\n")
            
            f.write("Cross-validation Results\n")
            f.write("------------------------\n")
            for i, acc in enumerate(fold_accuracies):
                f.write(f"Fold {i+1} accuracy: {acc:.4f}\n")
            
            f.write(f"\nAverage accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n\n")
            
            f.write("Per-class Performance\n")
            f.write("--------------------\n")
            for i in range(n_classes):
                f.write(f"{class_names[i]}:\n")
                f.write(f"  Total samples: {class_total[i]}\n")
                f.write(f"  Top-1 accuracy: {top1_accuracies[i]:.4f} ({class_top1_correct[i]}/{class_total[i]})\n")
                f.write(f"  Top-2 accuracy: {top2_accuracies[i]:.4f} ({class_top2_correct[i]}/{class_total[i]})\n\n")
            
            f.write("\nClassification Report\n")
            f.write("--------------------\n")
            f.write(classification_report(all_y_true, all_y_pred, target_names=class_names))
            
        # Create prediction function
        with open(os.path.join(output_dir, 'predict_cancer_type.py'), 'w') as f:
            f.write("import pickle\n")
            f.write("import numpy as np\n")
            f.write("import pandas as pd\n")
            f.write("import torch\n")
            f.write("import torch.nn as nn\n")
            f.write("import torch.nn.functional as F\n\n")
            
            # Write DualDecoderVAE class definition
            f.write("class DualDecoderVAE(nn.Module):\n")
            f.write("    def __init__(self, input_dim, latent_dim=8, hidden_dims=[64, 32], num_classes=7):\n")
            f.write("        super(DualDecoderVAE, self).__init__()\n")
            f.write("        self.input_dim = input_dim\n")
            f.write("        self.latent_dim = latent_dim\n")
            f.write("        self.num_classes = num_classes\n")
            f.write("        \n")
            f.write("        # Encoder\n")
            f.write("        encoder_layers = []\n")
            f.write("        prev_dim = input_dim\n")
            f.write("        for hidden_dim in hidden_dims:\n")
            f.write("            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))\n")
            f.write("            encoder_layers.append(nn.BatchNorm1d(hidden_dim))\n")
            f.write("            encoder_layers.append(nn.LeakyReLU(0.2))\n")
            f.write("            prev_dim = hidden_dim\n")
            f.write("        self.encoder = nn.Sequential(*encoder_layers)\n")
            f.write("        \n")
            f.write("        # Mean and variance for latent space\n")
            f.write("        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)\n")
            f.write("        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)\n")
            f.write("        \n")
            f.write("        # Decoder 1 - Reconstruction\n")
            f.write("        decoder_layers = []\n")
            f.write("        prev_dim = latent_dim\n")
            f.write("        for hidden_dim in reversed(hidden_dims):\n")
            f.write("            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))\n")
            f.write("            decoder_layers.append(nn.BatchNorm1d(hidden_dim))\n")
            f.write("            decoder_layers.append(nn.LeakyReLU(0.2))\n")
            f.write("            prev_dim = hidden_dim\n")
            f.write("        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))\n")
            f.write("        self.decoder = nn.Sequential(*decoder_layers)\n")
            f.write("        \n")
            f.write("        # Decoder 2 - Classification\n")
            f.write("        classifier_layers = []\n")
            f.write("        prev_dim = latent_dim\n")
            f.write("        for i, hidden_dim in enumerate(reversed(hidden_dims)):\n")
            f.write("            if i == 0:\n")
            f.write("                classifier_layers.append(nn.Linear(prev_dim, hidden_dim // 2))\n")
            f.write("                classifier_layers.append(nn.BatchNorm1d(hidden_dim // 2))\n")
            f.write("                classifier_layers.append(nn.LeakyReLU(0.2))\n")
            f.write("                prev_dim = hidden_dim // 2\n")
            f.write("            else:\n")
            f.write("                classifier_layers.append(nn.Linear(prev_dim, hidden_dim // 2))\n")
            f.write("                classifier_layers.append(nn.BatchNorm1d(hidden_dim // 2))\n")
            f.write("                classifier_layers.append(nn.LeakyReLU(0.2))\n")
            f.write("                prev_dim = hidden_dim // 2\n")
            f.write("        \n")
            f.write("        classifier_layers.append(nn.Linear(prev_dim, num_classes))\n")
            f.write("        self.classifier = nn.Sequential(*classifier_layers)\n")
            f.write("    \n")
            f.write("    def encode(self, x):\n")
            f.write("        hidden = self.encoder(x)\n")
            f.write("        mu = self.fc_mu(hidden)\n")
            f.write("        log_var = self.fc_var(hidden)\n")
            f.write("        return mu, log_var\n")
            f.write("    \n")
            f.write("    def reparameterize(self, mu, log_var):\n")
            f.write("        std = torch.exp(0.5 * log_var)\n")
            f.write("        eps = torch.randn_like(std)\n")
            f.write("        z = mu + eps * std\n")
            f.write("        return z\n")
            f.write("    \n")
            f.write("    def decode(self, z):\n")
            f.write("        return self.decoder(z)\n")
            f.write("    \n")
            f.write("    def classify(self, z):\n")
            f.write("        return self.classifier(z)\n")
            f.write("    \n")
            f.write("    def forward(self, x):\n")
            f.write("        mu, log_var = self.encode(x)\n")
            f.write("        z = self.reparameterize(mu, log_var)\n")
            f.write("        x_recon = self.decode(z)\n")
            f.write("        class_logits = self.classify(z)\n")
            f.write("        return x_recon, class_logits, mu, log_var\n")
            f.write("    \n")
            f.write("    def get_embeddings(self, x):\n")
            f.write("        self.eval()\n")
            f.write("        with torch.no_grad():\n")
            f.write("            mu, _ = self.encode(x)\n")
            f.write("        return mu\n\n")
            
            # Write prediction function
            f.write("def predict_cancer_type(sample_data, model_path='dualvae_multiclass_model.pt',\n")
            f.write("                       scaler_path='feature_scaler.pkl',\n")
            f.write("                       encoder_path='label_encoder.pkl',\n")
            f.write("                       feature_path='feature_list.txt',\n")
            f.write("                       top_k=2):\n")
            f.write("    \"\"\"\n")
            f.write("    Predict cancer type from sample data using the trained Dual VAE model.\n")
            f.write("    \n")
            f.write("    Parameters:\n")
            f.write("    -----------\n")
            f.write("    sample_data : pandas.DataFrame\n")
            f.write("        Sample data with features matching the training data\n")
            f.write("    model_path : str\n")
            f.write("        Path to the saved model\n")
            f.write("    scaler_path : str\n")
            f.write("        Path to the saved scaler\n")
            f.write("    encoder_path : str\n")
            f.write("        Path to the saved label encoder\n")
            f.write("    feature_path : str\n")
            f.write("        Path to the file containing feature names\n")
            f.write("    top_k : int, optional (default=2)\n")
            f.write("        Number of top predictions to return\n")
            f.write("        \n")
            f.write("    Returns:\n")
            f.write("    --------\n")
            f.write("    list\n")
            f.write("        List of dictionaries containing predictions for each sample\n")
            f.write("    \"\"\"\n")
            f.write("    # Load feature list\n")
            f.write("    with open(feature_path, 'r') as f:\n")
            f.write("        feature_cols = [line.strip() for line in f.readlines()]\n")
            f.write("    \n")
            f.write("    # Check that required features are present\n")
            f.write("    missing_features = [f for f in feature_cols if f not in sample_data.columns]\n")
            f.write("    if missing_features:\n")
            f.write("        raise ValueError(f'Missing features: {missing_features}')\n")
            f.write("    \n")
            f.write("    # Extract features\n")
            f.write("    X = sample_data[feature_cols].values\n")
            f.write("    \n")
            f.write("    # Handle NaN values\n")
            f.write("    X = np.nan_to_num(X, nan=np.nanmedian(X))\n")
            f.write("    \n")
            f.write("    # Load scaler\n")
            f.write("    with open(scaler_path, 'rb') as f:\n")
            f.write("        scaler = pickle.load(f)\n")
            f.write("    \n")
            f.write("    # Apply scaling\n")
            f.write("    X_scaled = scaler.transform(X)\n")
            f.write("    \n")
            f.write("    # Load label encoder\n")
            f.write("    with open(encoder_path, 'rb') as f:\n")
            f.write("        label_encoder = pickle.load(f)\n")
            f.write("    \n")
            f.write("    # Load model\n")
            f.write("    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n")
            f.write("    model = DualDecoderVAE(input_dim=len(feature_cols), num_classes=len(label_encoder.classes_))\n")
            f.write("    model.load_state_dict(torch.load(model_path, map_location=device))\n")
            f.write("    model.to(device)\n")
            f.write("    model.eval()\n")
            f.write("    \n")
            f.write("    # Convert to tensor\n")
            f.write("    X_tensor = torch.FloatTensor(X_scaled).to(device)\n")
            f.write("    \n")
            f.write("    # Get predictions\n")
            f.write("    with torch.no_grad():\n")
            f.write("        _, class_logits, embeddings, _ = model(X_tensor)\n")
            f.write("        probs = F.softmax(class_logits, dim=1).cpu().numpy()\n")
            f.write("        embeddings = embeddings.cpu().numpy()\n")
            f.write("    \n")
            f.write("    # Get top-k predictions for each sample\n")
            f.write("    results = []\n")
            f.write("    for i, p in enumerate(probs):\n")
            f.write("        top_indices = np.argsort(p)[-top_k:][::-1]\n")
            f.write("        top_probs = p[top_indices]\n")
            f.write("        top_labels = [label_encoder.classes_[idx] for idx in top_indices]\n")
            f.write("        \n")
            f.write("        results.append({\n")
            f.write("            'sample_idx': i,\n")
            f.write("            'top_labels': top_labels,\n")
            f.write("            'top_probabilities': top_probs,\n")
            f.write("            'embedding': embeddings[i]\n")
            f.write("        })\n")
            f.write("    \n")
            f.write("    return results\n")
            f.write("\n")
            f.write("if __name__ == '__main__':\n")
            f.write("    # Example usage\n")
            f.write("    print('Example usage:')\n")
            f.write("    print('1. Load your sample data')\n")
            f.write("    print('   sample_data = pd.read_csv(\"your_data.csv\")')\n")
            f.write("    print('2. Predict cancer types')\n")
            f.write("    print('   predictions = predict_cancer_type(sample_data)')\n")
            f.write("    print('3. View predictions')\n")
            f.write("    print('   for p in predictions:')\n")
            f.write("    print('       print(f\"Sample {p[\"sample_idx\"]}: {p[\"top_labels\"][0]} ({p[\"top_probabilities\"][0]:.2%})\")')\n")
        
        print(f"Prediction function saved to: {os.path.join(output_dir, 'predict_cancer_type.py')}")
        print("\nMulticlass cancer type prediction using Dual VAE completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting suggestions:")
        print("1. Make sure all dependencies are installed: pip install pandas numpy matplotlib seaborn torch scikit-learn umap-learn tqdm shap")
        print("2. Ensure 'PositiveCancerSEEKResults.xlsx' or 'Dataset.xlsx' is in the current directory")
        print("3. Check if your dataset has the 'Condition' column with cancer types")


if __name__ == "__main__":
    main()