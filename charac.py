import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style for plots
plt.style.use('ggplot')
sns.set_palette("Set2")
plt.rcParams.update({'font.size': 12})

# Read the Excel file
def analyze_cancer_data(file_path):
    # Read the Excel file
    try:
        df = pd.read_excel(file_path)
        print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        print("Column names:", df.columns.tolist())
        
        # Assuming there's a column called 'condition' that contains cancer types
        # Adjust the column name if it's different in your data
        condition_col = 'condition'
        
        # Check if the condition column exists
        if condition_col not in df.columns:
            possible_cols = [col for col in df.columns if 'cancer' in col.lower() 
                            or 'type' in col.lower() 
                            or 'condition' in col.lower() 
                            or 'diagnosis' in col.lower()]
            
            if possible_cols:
                print(f"'condition' column not found. Possible columns: {possible_cols}")
                condition_col = possible_cols[0]
                print(f"Using '{condition_col}' as the condition column")
            else:
                raise ValueError("Could not find a column for cancer types")
        
        # Create distribution plots
        generate_distribution_plots(df, condition_col)
        
        return df
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def generate_distribution_plots(df, condition_col):
    """Generate various distribution plots based on cancer type"""
    
    # 1. Simple count plot of cancer types
    plt.figure(figsize=(12, 6))
    counts = df[condition_col].value_counts()
    ax = sns.barplot(x=counts.index, y=counts.values)
    plt.title(f'Distribution of Cancer Types (n={len(df)})', fontsize=16)
    plt.xlabel('Cancer Type', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on top of each bar
    for i, count in enumerate(counts.values):
        ax.text(i, count + 0.1, str(count), ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('cancer_type_counts.png', dpi=300)
    plt.close()
    
    # 2. Percentage distribution pie chart
    plt.figure(figsize=(10, 10))
    percentages = 100 * counts / counts.sum()
    
    plt.pie(percentages, labels=counts.index, autopct='%1.1f%%', 
            startangle=90, shadow=False, textprops={'fontsize': 12})
    plt.axis('equal')
    plt.title('Percentage Distribution of Cancer Types', fontsize=16)
    plt.tight_layout()
    plt.savefig('cancer_type_percentages_pie.png', dpi=300)
    plt.close()
    
    # 3. Check for other demographic columns to create stratified distributions
    demo_cols = [col for col in df.columns if col.lower() in 
                ['age', 'gender', 'sex', 'stage', 'grade']]
    
    for col in demo_cols:
        if df[col].nunique() > 1 and df[col].nunique() <= 10:  # Only for categorical variables with reasonable number of categories
            plt.figure(figsize=(14, 8))
            
            # Create a crosstab to count cancer types by demographic variable
            cross_tab = pd.crosstab(df[condition_col], df[col])
            cross_tab.plot(kind='bar', stacked=True)
            
            plt.title(f'Cancer Types by {col}', fontsize=16)
            plt.xlabel('Cancer Type', fontsize=14)
            plt.ylabel('Count', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.legend(title=col)
            plt.tight_layout()
            plt.savefig(f'cancer_type_by_{col}.png', dpi=300)
            plt.close()
    
    # 4. Create a heatmap if there are stages or grades
    stage_cols = [col for col in df.columns if 'stage' in col.lower()]
    if stage_cols:
        stage_col = stage_cols[0]
        
        # Make sure stage is categorical/ordinal
        if df[stage_col].dtype == 'object' or df[stage_col].nunique() <= 5:
            plt.figure(figsize=(12, 8))
            
            # Create cross-tabulation
            stage_cancer_ct = pd.crosstab(df[stage_col], df[condition_col])
            
            # Calculate percentages within each cancer type
            stage_cancer_pct = stage_cancer_ct.div(stage_cancer_ct.sum(axis=0), axis=1) * 100
            
            # Create heatmap
            sns.heatmap(stage_cancer_pct, annot=True, cmap="YlGnBu", fmt='.1f')
            plt.title('Cancer Stage Distribution (%) by Type', fontsize=16)
            plt.ylabel('Stage', fontsize=14)
            plt.xlabel('Cancer Type', fontsize=14)
            plt.tight_layout()
            plt.savefig('cancer_stage_heatmap.png', dpi=300)
            plt.close()

# Run the analysis
if __name__ == "__main__":
    file_path = "PositiveCancerSEEKResults.xlsx"  # Update this path to your actual file location
    analyze_cancer_data(file_path)
    print("Analysis complete! Figures have been saved.")