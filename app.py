writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
with open('titanic_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Set page title
st.title('Titanic Survival Prediction')

# Sidebar navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select a page:', 
                          ['Data Exploration', 'Visualization', 'Prediction', 'Model Performance'])

# Data Exploration Section
if options == 'Data Exploration':
    st.header('Titanic Dataset Exploration')
    
    # Load data
    df = pd.read_csv('Titanic-Dataset.csv')
    
    st.subheader('Dataset Overview')
    st.write(f"Shape of the dataset: {df.shape}")
    st.write("First 5 rows:")
    st.write(df.head())
    
    st.subheader('Data Description')
    st.write(df.describe())
    
    st.subheader('Missing Values')
    st.write(df.isnull().sum())

# Visualization Section
elif options == 'Visualization':
    st.header('Data Visualizations')
    df = pd.read_csv('Titanic-Dataset.csv')
    
    # Survival by sex
    st.subheader('Survival by Sex')
    fig, ax = plt.subplots()
    sns.countplot(x='Sex', hue='Survived', data=df, ax=ax)
    st.pyplot(fig)
    
    # Survival by class
    st.subheader('Survival by Passenger Class')
    fig, ax = plt.subplots()
    sns.countplot(x='Pclass', hue='Survived', data=df, ax=ax)
    st.pyplot(fig)
    
    # Age distribution
    st.subheader('Age Distribution of Survivors vs Non-Survivors')
    fig, ax = plt.subplots(figsize=(10,6))
    sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True, ax=ax)
    st.pyplot(fig)

# Prediction Section
elif options == 'Prediction':
    st.header('Predict Survival on Titanic')
    
    # Input widgets
    pclass = st.selectbox('Passenger Class', [1, 2, 3])
    sex = st.selectbox('Sex', ['male', 'female'])
    age = st.slider('Age', 0, 100, 30)
    sibsp = st.slider('Number of Siblings/Spouses Aboard', 0, 8, 0)
    parch = st.slider('Number of Parents/Children Aboard', 0, 6, 0)
    fare = st.slider('Fare', 0, 600, 30)
    embarked = st.selectbox('Port of Embarkation', ['S', 'C', 'Q'])
    
    # Preprocess inputs
    sex = 0 if sex == 'male' else 1
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    embarked = embarked_map[embarked]
    
    # Make prediction
    if st.button('Predict Survival'):
        input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)
        
        if prediction[0] == 1:
            st.success('This passenger would have survived!')
        else:
            st.error('This passenger would not have survived.')
            
        st.write(f"Probability of survival: {probability[0][1]:.2f}")

# Model Performance Section
elif options == 'Model Performance':
    st.header('Model Performance Metrics')
    
    st.subheader('Random Forest Classifier')
    st.write("""
    - Accuracy: ~80%
    - Good for handling non-linear relationships
    - Less prone to overfitting
    """)
    
    st.subheader('Feature Importance')
    # Load model to show feature importance
    with open('titanic_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    importance = model.feature_importances_
    
    fig, ax = plt.subplots()
    sns.barplot(x=importance, y=features, ax=ax)
    ax.set_title('Feature Importance')
    st.pyplot(fig)
