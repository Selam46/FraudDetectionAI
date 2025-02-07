# Fraud Detection Analysis

## Task 1: Data Analysis and Preprocessing

This project focuses on analyzing and visualizing fraud detection data using Python. In Task 1, we perform a series of data preprocessing and exploratory data analysis (EDA) steps to prepare the dataset for further analysis.

### Description

In Task 1, the following steps are executed:

1. **Handle Missing Values**:
   - Impute or drop missing values to ensure data quality.

2. **Data Cleaning**:
   - Remove duplicate entries to avoid skewed results.
   - Correct data types for consistency and accuracy.

3. **Exploratory Data Analysis (EDA)**:
   - Conduct univariate analysis to understand individual features.
   - Perform bivariate analysis to explore relationships between features.

4. **Merge Datasets for Geolocation Analysis**:
   - Convert IP addresses to integer format for easier processing.
   - Merge `Fraud_Data.csv` with `IpAddress_to_Country.csv` to include geolocation data.

5. **Feature Engineering**:
   - Calculate transaction frequency and velocity from `Fraud_Data.csv`.
   - Create time-based features, including:
     - `hour_of_day`
     - `day_of_week`

6. **Normalization and Scaling**:
   - Normalize and scale features to ensure they are on a similar scale.

7. **Encode Categorical Features**:
   - Apply encoding techniques to convert categorical features into numerical format for analysis.

### Project Structure

FraudDetectionAI/
│
├── data/
│   ├── raw/
│   │   ├── Fraud_Data.csv
│   │   └── IpAddress_to_Country.csv
│   └── processed/
│       ├── train_data.csv
│       └── test_data.csv
│
├── notebooks/
│   ├── 01_data_analysis_preprocessing.ipynb
│   └── 02_model_preparation.ipynb
│
├── src/
│   ├── data_cleaning.py
│   └── model_utils.py
│
├── requirements.txt
└── README.md

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Selam46/FraudDetectionAI.git
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

