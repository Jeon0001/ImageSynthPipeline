import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def calculate_metrics(file_path):
    # 1. Load the data
    data = pd.read_csv(file_path)
    
    # 2. Create two “Match” columns:
    #    - Match if `original_country` is mentioned in the `response`
    #    - Match if `label` is mentioned in the `response`
    
    data['Match_country'] = data.apply(
        lambda row: 1 if any(term.lower() in row['response'].lower() 
                             for term in row['original_country'].split(',')) 
                   else 0, 
        axis=1
    )
    
    data['Match_label'] = data.apply(
        lambda row: 1 if any(term.lower() in row['response'].lower() 
                             for term in row['label'].split(',')) 
                   else 0, 
        axis=1
    )

    # 3. Normalize the text of both `original_country` and `label`
    #    to ensure consistent grouping (e.g., remove duplicates, sort, etc.)
    
    data['original_country'] = data['original_country'].apply(
        lambda x: ', '.join(sorted(set(term.strip() for term in x.split(','))))
    )
    
    data['label'] = data['label'].apply(
        lambda x: ', '.join(sorted(set(term.strip() for term in x.split(','))))
    )
    
    # 4. Compute metrics for `original_country`
    country_results = []
    unique_countries = data['original_country'].unique()
    
    for country in unique_countries:
        country_data = data[data['original_country'] == country]
        y_true = [1] * len(country_data)  # ground truth: should mention the country
        y_pred = country_data['Match_country'].tolist()
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)
        accuracy  = accuracy_score(y_true, y_pred)
        
        # Identify misclassified samples (for debugging or inspection)
        misclassified_indices = [
            i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred
        ]
        misclassified_data = country_data.iloc[misclassified_indices]
        
        print(f"[ORIGINAL_COUNTRY] {country}")
        print("Indices of misclassified samples:", misclassified_indices)
        print("Misclassified data:\n", misclassified_data)
        print("=======================================================")
        
        country_results.append({
            'Country': country,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Accuracy': accuracy,
            'Correct Samples': sum(y_pred),    # sum of 1s
            'Total Samples': len(country_data) # total rows in this group
        })
    
    # Convert country results to a DataFrame
    country_results_df = pd.DataFrame(country_results)
    
    # 5. Compute metrics for `label`
    label_results = []
    unique_labels = data['label'].unique()
    
    for lbl in unique_labels:
        label_data = data[data['label'] == lbl]
        y_true = [1] * len(label_data)  # ground truth: should mention the label
        y_pred = label_data['Match_label'].tolist()
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)
        accuracy  = accuracy_score(y_true, y_pred)
        
        # Identify misclassified samples
        misclassified_indices = [
            i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred
        ]
        misclassified_data = label_data.iloc[misclassified_indices]
        
        print(f"[LABEL] {lbl}")
        print("Indices of misclassified samples:", misclassified_indices)
        print("Misclassified data:\n", misclassified_data)
        print("=======================================================")
        
        label_results.append({
            'Label': lbl,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Accuracy': accuracy,
            'Correct Samples': sum(y_pred),
            'Total Samples': len(label_data)
        })
    
    # Convert label results to a DataFrame
    label_results_df = pd.DataFrame(label_results)
    
    # 6. Return both results
    return country_results_df, label_results_df


# Showing the results
file_path = r'C:\Users\Admin\Downloads\Myanmar_Original_Food_Results.csv'
country_df, label_df = calculate_metrics(file_path)
print("=== Metrics By Original Country ===")
print(country_df)
print("\n=== Metrics By Label ===")
print(label_df)
