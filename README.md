# Machine Learning Model Deployment with Streamlit

This project demonstrates how to deploy a machine learning model using Streamlit. The example uses the Titanic dataset to predict passenger survival.

## Features
- Data exploration and visualization
- Model training and evaluation
- Interactive web app for predictions

## Project Structure
```
├── app.py                  # Streamlit app
├── requirements.txt        # Python dependencies
├── titanic_model.pkl       # Trained ML model
├── Titanic-Dataset.csv     # Main dataset
├── Titanic-Dataset (1).csv # Additional dataset
├── notebooks/              # Jupyter notebooks for EDA/modeling
└── README.md               # Project documentation
```

## Getting Started

### 1. Clone the repository
```
git clone <repo-url>
cd Machine-Learning-Model-Deployment-with-Streamlit
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```
streamlit run app.py
```

## Usage
- Open the app in your browser (Streamlit will provide a local URL).
- Upload or use the provided Titanic dataset.
- Explore data, visualize features, and make predictions.

## Requirements
- Python 3.7+
- See `requirements.txt` for all dependencies

## Files
- `app.py`: Main Streamlit application
- `titanic_model.pkl`: Pre-trained model (pickle file)
- `Titanic-Dataset.csv`: Titanic dataset for predictions
- `notebooks/`: Contains Jupyter notebooks for data analysis and model building

## License
This project is for educational purposes.
