# Smart Lender

Smart Lender is a Flask-based machine learning project for predicting loan eligibility using structured borrower data.

## Features
- Decision Tree, Random Forest, KNN, and XGBoost model training
- Data preprocessing and exploratory data analysis helpers
- Model comparison and selection
- Flask web UI and JSON APIs
- Prediction history stored in JSON
- Validation and error handling

## Project Structure
- `app.py` - Flask application
- `train_model.py` - training entrypoint
- `src/ml_pipeline.py` - preprocessing, training, prediction, history helpers
- `loan_prediction.csv` - dataset (place your file in the project root)
- `templates/` - Jinja templates
- `static/` - CSS, JS, images

## Setup
1. Create and activate a virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Place `loan_prediction.csv` in the project root
4. Train the model: `python train_model.py`
5. Run the app: `python app.py`

## Routes
- `/` - Home page
- `/predict` - Loan prediction form
- `/history` - Prediction history page
- `/about` - Project overview
- `/contact` - Contact page
- `/api/predict` - JSON prediction API
- `/api/history` - JSON history API
- `/api/history/<id>` - History detail API
