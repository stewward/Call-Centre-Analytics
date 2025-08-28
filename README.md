# Call Centre Analytics Project

## Overview
This project analyzes simulated call centre data to uncover trends in daily call volume and service length. 
Using Python, pandas, and statsmodels. Evaluating and exploring data analysis, visualized call trends, and applied forecasting and predictive modeling techniques.  

---

## Dataset
- **File:** `simulated_call_centre.csv`  
- **Description:** Simulated call centre data including:
  - `call_id` – unique call identifier
  - `date` – date of the call
  - `daily_caller` – number of callers per day
  - `call_started`, `call_answered`, `call_ended` – timestamps
  - `wait_length`, `service_length` – call metrics in seconds
  - `meets_standard` – whether service meets target standards

## Key Steps
1. **Data Cleaning & Inspection**
   - Checked for missing values
   - Verified data types
   - Converted `date` to datetime format

2. **Exploratory Data Analysis (EDA)**
   - Calculated daily call volume and average service length
   - Plotted trends over time using Matplotlib

3. **Forecasting**
   - Predicted daily call volume for the next week using `ExponentialSmoothing` from `statsmodels`
   - Evaluated trends and patterns for operational insights

4. **Predictive Modeling**
   - Attempted to predict `service_length` using `wait_length` with linear regression (R² ≈ 0)
   - Demonstrates workflow for predictive modeling and feature selection

## Libraries Used
- `pandas` for data manipulation
- `numpy` for numerical operations
- `matplotlib` for visualization
- `statsmodels` for time series forecasting
- `scikit-learn` for regression


## Usage
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/Call-Centre-Analytics.git
