# Telecom Customer Churn Prediction

A Flask web application for predicting whether a telecom customer is likely to churn, based on customer usage and plan data.

## Project Overview

This project uses a pre-trained Random Forest classification model to predict customer churn. Users can input key telecom customer features through a web interface, and the app returns a churn prediction.

## Folder Structure

- `app/` - Main Flask application directory
  - `app.py.py` - Flask application script
  - `models/` - Saved model and preprocessing objects
  - `templates/` - HTML templates for the web UI
  - `static/` - Static files such as CSS
- `data/` - Raw and processed dataset files
- `notebook/` - Jupyter notebook for model exploration and development
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.10+ (recommended)
- Packages listed in `requirements.txt`

## Installation

1. Create a virtual environment (recommended):

```bash
python -m venv venv
```

2. Activate the virtual environment:

- Windows PowerShell:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- Windows Command Prompt:
  ```cmd
  .\venv\Scripts\activate.bat
  ```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

1. Navigate to the app directory:

```bash
cd app
```

2. Run the Flask app:

```bash
python app.py.py
```

3. Open your browser and go to:

```text
http://127.0.0.1:5000/
```

## Using the App

On the web page, fill in the following fields:

- International Plan
- Voice Mail Plan
- Account Length
- Number of Voicemail Messages
- Total Day Minutes
- Total Day Calls
- Total Evening Minutes
- Total Evening Calls
- Total Night Minutes
- Total Night Calls
- Total International Minutes
- Total International Calls
- Customer Service Calls

Click **Predict** to see whether the model estimates the customer will churn or not.

## Notes

- The application uses `random_forest_model.pkl` in `app/models/`.
- Categorical features are encoded with the saved `label_encoder.pkl`.
- Numeric values are scaled using `standard_scaler.pkl` before prediction.

## Optional Improvements

- Add input validation for numeric fields.
- Expose model performance metrics and explanation details.
- Use a production-ready WSGI server for deployment.
