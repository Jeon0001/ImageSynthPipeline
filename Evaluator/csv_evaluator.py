import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Load the CSV file
def calculate_metrics(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Add a column to check if Original_Country is mentioned in Content
    data['Match'] = data.apply(
        lambda row: 1 if any(term.lower() in row['response'].lower() for term in row['original_country'].split(',')) else 0, axis=1
    )

    # Normalize Original_Country to treat each unique combination as a single entity
    data['original_country'] = data['original_country'].apply(
        lambda x: ', '.join(sorted(set(term.strip() for term in x.split(','))))
    )

    # Group the data by original_country and calculate metrics for each group
    results = []
    unique_countries = data['original_country'].unique()

    for country in unique_countries:
        country_data = data[data['original_country'] == country]
        y_true = [1] * len(country_data)  # Ground truth is always positive for this task
        y_pred = country_data['Match'].tolist()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        results.append({
            'Country': country,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Accuracy': accuracy,
            'Correct Samples': sum(y_pred),
            'Total Samples': len(country_data)
        })

    # Convert results to a DataFrame for easy viewing
    results_df = pd.DataFrame(results)
    return results_df

# Specify the path to your file
# file_path = 'D:/Data Downloads/Bing Image Scraped Results/responses_synthesized.csv'
file_path = 'responses/responses_synthesized.csv'

# Calculate metrics
results_df = calculate_metrics(file_path)

# Display the results
print(results_df)