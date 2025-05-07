# main_enhanced.py
from data_processing import process_dass_data
from scoring import compute_change_scores
from statis import display_statistical_results
from visualization import run_visualizations
import pandas as pd

def main():
    """
    Main function to process DASS-21 data, run statistical analysis,
    and display results in the exact format requested.
    """
    try:
        # Path to your Excel file - update this to your actual file path
        file_path =r"C:\Nistha\DASS-1\datasets\DASS21_Scaled_Final_Data.xlsx"
        
        # Load and process data
        pre = process_dass_data(file_path, 'Pre-Test')
        post = process_dass_data(file_path, 'Post-Test')
        
        # Merge datasets
        merged = pd.merge(pre, post, on='Name', suffixes=('_pre', '_post'))
        merged = compute_change_scores(merged)
        
        # Run and display statistical results in the requested format
        display_statistical_results(merged)

        #visulaosation
        run_visualizations(merged)
        
    except FileNotFoundError:
        print(f"Error: Excel file not found. Please check the file path.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()