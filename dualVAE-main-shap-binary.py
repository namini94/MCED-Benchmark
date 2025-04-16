import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, classification_report, confusion_matrix,
    recall_score, precision_score
)
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
        if col not in ['PatientID', 'SampleID', 'Condition', 'Stage']:
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

def create_binary_labels(data, normal_condition='Normal'):
    binary_labels = data['Condition'].apply(lambda x: 0 if x == normal_condition else 1)
    normal_count = sum(binary_labels == 0)
    cancer_count = sum(binary_labels == 1)
    print(f"Binary classification: {normal_count} Normal samples, {cancer_count} Cancer samples")
    return binary_labels

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
            scores.append(probs[:, 1].cpu().numpy())  # Prob of class 1 (Cancer)
    
    return np.concatenate(predictions), np.concatenate(scores)

# New visualization functions based on XGBoost code

def plot_training_losses(losses, output_dir='dual_vae_results', filename='training_losses.pdf'):
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

def plot_fold_accuracies(fold_accuracies, output_dir='dual_vae_results', filename='fold_accuracies.pdf'):
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

def plot_roc_curve(y_true, y_scores, output_dir='dual_vae_results', filename='roc_curve.pdf'):
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
        
    Returns:
    --------
    tuple
        (AUC score, threshold for 99% specificity)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find threshold for >99% specificity (similar to the red dot in the figure)
    specificity = 1 - fpr
    high_spec_indices = np.where(specificity >= 0.99)[0]
    if len(high_spec_indices) > 0:
        high_spec_index = high_spec_indices[-1]
        high_spec_tpr = tpr[high_spec_index]
        high_spec_fpr = fpr[high_spec_index]
        high_spec_threshold = thresholds[high_spec_index]
    else:
        # Fallback if no point with >99% specificity
        high_spec_index = np.argmax(specificity)
        high_spec_tpr = tpr[high_spec_index]
        high_spec_fpr = fpr[high_spec_index]
        high_spec_threshold = thresholds[high_spec_index]
    
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
             label=f'Sensitivity: {high_spec_tpr:.0%} at >99% specificity (threshold: {high_spec_threshold:.3f})')
    
    # Set axis labels and style
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve for Dual Decoder VAE', fontsize=18, fontweight='bold')
    
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
    
    # Also create the alternate visualization with specificity on x-axis
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
             label=f'Sensitivity: {high_spec_tpr:.0%} at >99% specificity (threshold: {high_spec_threshold:.3f})')
    
    # Set axis labels and style
    plt.xlabel('Specificity (%)', fontsize=14)
    plt.ylabel('Sensitivity (%)', fontsize=14)
    plt.title('ROC Curve for Dual Decoder VAE (Alternate Format)', fontsize=18, fontweight='bold')
    
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
    
    return roc_auc, high_spec_threshold

def plot_sensitivity_by_cancer_type(cancer_sensitivities, threshold, output_dir='dual_vae_results', filename='sensitivity_by_cancer_type.pdf'):
    """
    Plot sensitivity by cancer type.
    
    Parameters:
    -----------
    cancer_sensitivities : dict
        Dictionary with cancer type as key and sensitivity data as value
    threshold : float
        The classification threshold used (for 99% specificity)
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
    
    # Set labels and title with threshold information
    plt.xlabel('Cancer Type', fontsize=14)
    plt.ylabel('Proportion detected\nby Dual VAE (%)', fontsize=14)
    plt.title('Sensitivity of Dual Decoder VAE by tumor type)', 
              fontsize=16, fontweight='bold')
    
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

def plot_sensitivity_by_stage(stage_sensitivities, threshold, output_dir='dual_vae_results', filename='sensitivity_by_stage.pdf'):
    """
    Plot sensitivity by cancer stage.
    
    Parameters:
    -----------
    stage_sensitivities : dict
        Dictionary with stage as key and sensitivity as value
    threshold : float
        The classification threshold used (for 99% specificity)
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
    
    # Set labels and title with threshold information
    plt.xlabel('Stage', fontsize=14)
    plt.ylabel('Proportion detected\nby Dual VAE (%)', fontsize=14)
    plt.title('Sensitivity of Dual Decoder VAE by stage)', 
              fontsize=16, fontweight='bold')
    
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

def plot_latent_space(embeddings, labels, output_dir='dual_vae_results', filename='latent_space.pdf'):
    """
    Plot UMAP visualization of the latent space.
    
    Parameters:
    -----------
    embeddings : numpy.ndarray
        Latent space embeddings
    labels : numpy.ndarray
        Class labels
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
    
    plt.figure(figsize=(12, 10))
    colors = ['#3498db', '#e74c3c']  # Blue for Normal, Red for Cancer
    
    for i, label_name in enumerate(['Normal', 'Cancer']):
        idx = labels == i
        plt.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1], color=colors[i], 
                    label=label_name, alpha=0.7, edgecolors='none', s=50)
    
    plt.title('UMAP Visualization of Dual VAE Latent Space', fontsize=18, fontweight='bold')
    plt.xlabel('UMAP 1', fontsize=14)
    plt.ylabel('UMAP 2', fontsize=14)
    plt.legend(title='Class', title_fontsize=12, fontsize=12, markerscale=1.5)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_feature_importance_shap(model, X_data, feature_names, output_dir='dual_vae_results', n_background=100):
    """
    Analyze feature importance using SHAP values.
    
    Parameters:
    -----------
    model : DualDecoderVAE
        Trained Dual Decoder VAE model
    X_data : numpy.ndarray
        Data for explanation
    feature_names : list
        List of feature names
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
        return probas[:,1]  # Return probability of cancer class
    
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
    
    # Calculate mean absolute SHAP values for ranking features
    mean_abs_shap = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(feature_names, mean_abs_shap)), 
                                      columns=['Feature','SHAP Importance'])
    feature_importance.sort_values(by=['SHAP Importance'], ascending=False, inplace=True)
    
    # Plot feature importance as a bar chart
    plt.figure(figsize=(12, 10))
    plt.barh(feature_importance['Feature'][:20], feature_importance['SHAP Importance'][:20])
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Top 20 Features Ranked by SHAP Importance')
    plt.gca().invert_yaxis()  # Invert y-axis to have highest importance at the top
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_feature_importance_bar.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Use SHAP's summary plot with bar type
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values, samples, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Cancer Class)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_feature_importance.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot SHAP summary plot (dot plot) for top 20 features
    plt.figure(figsize=(14, 12))
    shap.summary_plot(shap_values, samples, feature_names=feature_names, max_display=20, show=False)
    plt.title("SHAP Values for Top 20 Features (Cancer Class)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shap_top_features.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save top 20 features to a text file
    with open(os.path.join(output_dir, 'top_features_by_shap.txt'), 'w') as f:
        f.write("Rank\tFeature\tSHAP Importance\n")
        for i, (feature, importance) in enumerate(zip(feature_importance['Feature'][:20], 
                                                     feature_importance['SHAP Importance'][:20])):
            f.write(f"{i+1}\t{feature}\t{importance:.6f}\n")
    
    # Create waterfall plots for the top 5 features
    top_features = feature_importance['Feature'][:5].tolist()
    top_indices = [list(feature_names).index(feat) for feat in top_features]
    
    # Find samples with the highest absolute SHAP values for each top feature
    for i, feat_idx in enumerate(top_indices):
        feature_name = feature_names[feat_idx]
        # Get sample with highest absolute SHAP value for this feature
        sample_idx = np.argmax(np.abs(shap_values[:, feat_idx]))
        
        plt.figure(figsize=(12, 8))
        shap.plots.waterfall(shap.Explanation(values=shap_values[sample_idx], 
                                           base_values=explainer.expected_value,
                                           data=samples[sample_idx],
                                           feature_names=feature_names),
                           max_display=10, show=False)
        plt.title(f"SHAP Waterfall Plot for Top Sample: {feature_name}", fontsize=16)
        plt.tight_layout()
        safe_feature_name = feature_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(output_dir, f'shap_waterfall_{safe_feature_name}.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        
    # Create dependence plots for the top 5 features
    for i, feat_idx in enumerate(top_indices):
        feature_name = feature_names[feat_idx]
        
        plt.figure(figsize=(12, 8))
        shap.dependence_plot(feat_idx, shap_values, samples, feature_names=feature_names, show=False)
        plt.title(f"SHAP Dependence Plot: {feature_name}", fontsize=16)
        plt.tight_layout()
        safe_feature_name = feature_name.replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(output_dir, f'shap_dependence_{safe_feature_name}.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
    
def main():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        output_dir = 'dual_vae_results'
        os.makedirs(output_dir, exist_ok=True)
        
        print("Reading data...")
        data = pd.read_excel("Dataset.xlsx")
        
        if 'Condition' not in data.columns:
            raise ValueError("Dataset must contain a 'Condition' column with cancer types")
        
        print(f"Dataset shape: {data.shape}")
        print(f"Cancer types: {data['Condition'].unique()}")
        print(f"Number of cancer types: {data['Condition'].nunique()}")
        
        print("Cleaning data (removing * characters)...")
        data_cleaned = clean_numeric_data(data)
        
        available_proteins = [protein for protein in PROTEIN_FEATURES if protein in data_cleaned.columns]
        
        omega_cols = [col for col in data_cleaned.columns if 'Omega' in col]
        
        if omega_cols:
            print(f"Found mutation feature: {omega_cols}")
        else:
            print("No Omega score column found. Model will only use protein features.")
            omega_cols = []
        
        feature_cols = available_proteins + omega_cols
        
        print("Applying log2 normalization...")
        data_log2 = log2_normalize(data_cleaned)
        
        print("Removing rows with NaN values...")
        data_clean = remove_nan_rows(data_log2, feature_cols)
        
        # Create binary labels and ADD AS A COLUMN to the dataframe 
        binary_labels = create_binary_labels(data_clean)
        data_clean['binary_label'] = binary_labels.values
        
        X = data_clean[feature_cols].values
        y = binary_labels.values
        
        print(f"Number of features: {X.shape[1]}")
        print(f"Number of samples after cleaning: {X.shape[0]}")
        print(f"Number of classes: 2 (Normal vs. Cancer)")
        
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values still present in data after cleaning.")
            print("Imputing remaining NaN values with feature medians...")
            imputer = SimpleImputer(strategy='median')
            X = imputer.fit_transform(X)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        latent_dim = 8
        hidden_dims = [64, 32]
        
        n_splits = 10
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        
        all_embeddings = []
        all_labels = []
        fold_metrics = []
        all_y_true = []
        all_y_pred = []
        all_y_scores = []
        fold_accuracies = []
        
        # Track sensitivity by stage
        stages = data_clean['Stage'].dropna().unique() if 'Stage' in data_clean.columns else []
        stage_predictions = {stage: {'correct': 0, 'total': 0} for stage in stages}
        
        # Track sensitivity by cancer type
        cancer_types = [ct for ct in data_clean['Condition'].unique() if ct != 'Normal']
        cancer_predictions = {ct: {'correct': 0, 'total': 0} for ct in cancer_types}
        
        # Store models and test loaders for later use with high specificity threshold
        models = []
        test_loaders = []
        fold_indices = []
        
        print(f"\nPerforming {n_splits}-fold cross-validation...")
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled, y)):
            print(f"\nFold {fold+1}/{n_splits}")
            
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            X_train_tensor = torch.FloatTensor(X_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_train_tensor = torch.LongTensor(y_train)
            y_test_tensor = torch.LongTensor(y_test)
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            input_dim = X_train.shape[1]
            model = DualDecoderVAE(input_dim=input_dim, latent_dim=latent_dim, 
                                  hidden_dims=hidden_dims, num_classes=2).to(device)
            
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
                lambda_cls=20.0,
                scheduler=scheduler
            )
            
            plot_training_losses(
                losses=losses,
                output_dir=output_dir,
                filename=f'training_losses_fold_{fold+1}.pdf'
            )
            
            # Get predictions from the built-in classifier
            y_pred, y_scores = get_classifier_predictions(model, test_loader, device)
            
            # Calculate metrics
            accuracy = np.mean(y_pred == y_test)
            fold_accuracies.append(accuracy)
            
            # Get embeddings for visualization
            test_embeddings = get_vae_embeddings(model, test_loader, device)
            if np.isnan(test_embeddings).any():
                print("Warning: NaN values in embeddings. Replacing with zeros.")
                test_embeddings = np.nan_to_num(test_embeddings)
            
            # Track predictions for cancer types and stages
            test_samples = data_clean.iloc[test_idx]
            test_indices_map = {idx: i for i, idx in enumerate(test_idx)}
            
            # Track sensitivity by stage if stage information is available
            if 'Stage' in data_clean.columns:
                for stage in stages:
                    stage_samples = test_samples[test_samples['Stage'] == stage]
                    for idx in stage_samples.index:
                        if idx in test_indices_map and test_samples.loc[idx, 'binary_label'] == 1:  # Only count cancer samples
                            stage_predictions[stage]['total'] += 1
                            rel_idx = test_indices_map[idx]
                            if y_pred[rel_idx] == 1:  # Correctly predicted as cancer
                                stage_predictions[stage]['correct'] += 1
            
            # Track sensitivity by cancer type
            for ct in cancer_types:
                ct_samples = test_samples[test_samples['Condition'] == ct]
                for idx in ct_samples.index:
                    if idx in test_indices_map:
                        cancer_predictions[ct]['total'] += 1
                        rel_idx = test_indices_map[idx]
                        if y_pred[rel_idx] == 1:  # Correctly predicted as cancer
                            cancer_predictions[ct]['correct'] += 1
            
            # Store results for later visualization
            all_embeddings.append(test_embeddings)
            all_labels.append(y_test)
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred)
            all_y_scores.extend(y_scores)
            
            # Store model, test loader, and test indices for later use with high specificity threshold
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
        
        # Combine results from all folds
        combined_embeddings = np.vstack(all_embeddings)
        combined_labels = np.concatenate(all_labels)
        combined_y_true = np.array(all_y_true)
        combined_y_pred = np.array(all_y_pred)
        combined_y_scores = np.array(all_y_scores)
        
        print("\nPlotting ROC curve...")
        roc_auc, high_spec_threshold = plot_roc_curve(combined_y_true, combined_y_scores, output_dir)
        print(f"Threshold for 99% specificity: {high_spec_threshold:.4f}")
        
        # Clear and recalculate sensitivities with the high specificity threshold
        stage_predictions = {stage: {'correct': 0, 'total': 0} for stage in stages}
        cancer_predictions = {ct: {'correct': 0, 'total': 0} for ct in cancer_types}
        
        # Recalculate stage and cancer type sensitivities with high specificity threshold
        for fold, (train_idx, test_idx) in enumerate(fold_indices):
            y_test = y[test_idx]
            test_samples = data_clean.iloc[test_idx]
            test_indices_map = {idx: i for i, idx in enumerate(test_idx)}
            
            # Get scores for this fold's test set
            _, y_scores = get_classifier_predictions(models[fold], test_loaders[fold], device)
            
            # Apply the high specificity threshold
            y_pred_high_spec = (y_scores >= high_spec_threshold).astype(int)
            
            # Track sensitivity by stage with high specificity threshold
            if 'Stage' in data_clean.columns:
                for stage in stages:
                    stage_samples = test_samples[test_samples['Stage'] == stage]
                    for idx in stage_samples.index:
                        if idx in test_indices_map and test_samples.loc[idx, 'binary_label'] == 1:
                            stage_predictions[stage]['total'] += 1
                            rel_idx = test_indices_map[idx]
                            if y_pred_high_spec[rel_idx] == 1:  # Using high spec threshold
                                stage_predictions[stage]['correct'] += 1
            
            # Track sensitivity by cancer type with high specificity threshold
            for ct in cancer_types:
                ct_samples = test_samples[test_samples['Condition'] == ct]
                for idx in ct_samples.index:
                    if idx in test_indices_map:
                        cancer_predictions[ct]['total'] += 1
                        rel_idx = test_indices_map[idx]
                        if y_pred_high_spec[rel_idx] == 1:  # Using high spec threshold
                            cancer_predictions[ct]['correct'] += 1
        
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
            plot_sensitivity_by_stage(stage_sensitivities, high_spec_threshold, output_dir)
        
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
            plot_sensitivity_by_cancer_type(cancer_sensitivities, high_spec_threshold, output_dir)
        
        # Visualize latent space
        print("\nVisualizing latent space...")
        plot_latent_space(combined_embeddings, combined_labels, output_dir)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(combined_y_true, combined_y_pred, target_names=['Normal', 'Cancer']))
        
        # Save confusion matrix
        conf_matrix = confusion_matrix(combined_y_true, combined_y_pred)
        print("\nConfusion Matrix:")
        print(conf_matrix)
        
        # Add after plotting the ROC curve and before training the final model
        print("\nPerforming SHAP analysis for feature importance...")
        best_fold_idx = np.argmax([fold_accuracies[i] for i in range(len(fold_accuracies))])
        print(f"Using model from best fold (Fold {best_fold_idx+1}) for SHAP analysis")

        best_model = models[best_fold_idx]  # Use the stored model from the best fold

        # Analyze feature importance using SHAP
        analyze_feature_importance_shap(
            model=best_model,
            X_data=X_scaled,
            feature_names=feature_cols,
            output_dir=output_dir,
            n_background=100
        )
        # Save the final model
        final_model = DualDecoderVAE(
            input_dim=X_scaled.shape[1], 
            latent_dim=latent_dim, 
            hidden_dims=hidden_dims, 
            num_classes=2
        ).to(device)
        
        # Train on all data
        all_tensor_X = torch.FloatTensor(X_scaled).to(device)
        all_tensor_y = torch.LongTensor(y).to(device)
        all_dataset = TensorDataset(all_tensor_X, all_tensor_y)
        all_loader = DataLoader(all_dataset, batch_size=64, shuffle=True)
        
        optimizer = optim.Adam(final_model.parameters(), lr=1e-3, weight_decay=1e-5)
        
        print("\nTraining final model on all data...")
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
        model_path = os.path.join(output_dir, 'dualvae_model.pt')
        torch.save(final_model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        
        # Save scaler for preprocessing new data
        scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Scaler saved to: {scaler_path}")
        
        # Save results to a text file
        with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
            f.write("Dual Decoder VAE with Integrated Classification for Normal vs. Cancer Classification\n")
            f.write("=============================================================================\n\n")
            
            f.write(f"Original dataset shape: {data.shape}\n")
            f.write(f"Final dataset shape after cleaning: {X.shape}\n")
            f.write(f"Number of features: {X.shape[1]}\n")
            f.write(f"Classification task: Binary (Normal vs. Cancer)\n\n")
            
            f.write(f"VAE latent dimension: {latent_dim}\n")
            f.write(f"KL divergence weight (beta): 5.0\n")
            f.write(f"Classification loss weight (lambda): 20.0\n\n")
            
            f.write("Cross-validation Results\n")
            f.write("------------------------\n")
            for i, acc in enumerate(fold_accuracies):
                f.write(f"Fold {i+1} accuracy: {acc:.4f}\n")
            
            f.write(f"\nAverage accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n\n")
            
            f.write("ROC AUC Score: {:.4f}\n\n".format(roc_auc))
            
            f.write("Sensitivity by Cancer Type\n")
            f.write("--------------------------\n")
            for ct, data in cancer_sensitivities.items():
                f.write(f"{ct}: {data['sensitivity']:.4f} ± {data['error']:.4f}\n")
            
            if stage_sensitivities:
                f.write("\nSensitivity by Stage\n")
                f.write("-------------------\n")
                for stage, data in stage_sensitivities.items():
                    f.write(f"Stage {stage}: {data['sensitivity']:.4f} ± {data['error']:.4f}\n")
            
            f.write("\nConfusion Matrix\n")
            f.write("---------------\n")
            f.write(f"True Negative: {conf_matrix[0, 0]}\n")
            f.write(f"False Positive: {conf_matrix[0, 1]}\n")
            f.write(f"False Negative: {conf_matrix[1, 0]}\n")
            f.write(f"True Positive: {conf_matrix[1, 1]}\n\n")
            
            f.write("Classification Report\n")
            f.write("--------------------\n")
            f.write(classification_report(combined_y_true, combined_y_pred, target_names=['Normal', 'Cancer']))
        
        print(f"\nAnalysis complete! All visualizations and results saved to '{output_dir}' directory.")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nTroubleshooting suggestions:")
        print("1. Make sure all dependencies are installed: pip install pandas numpy matplotlib seaborn torch scikit-learn umap-learn tqdm")
        print("2. Ensure 'Dataset.xlsx' is in the current directory")
        print("3. Check if your dataset has the 'Condition' column with cancer types")


if __name__ == "__main__":
    main()