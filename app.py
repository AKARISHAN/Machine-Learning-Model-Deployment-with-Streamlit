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

# Prediction Section (FIXED VERSION)
elif options == 'Prediction':
    st.header('Predict Survival on Titanic')
    
    # Input widgets
    st.subheader('Passenger Details')
    col1, col2 = st.columns(2)
    with col1:
        pclass = st.selectbox('Passenger Class', [1, 2, 3])
        sex = st.selectbox('Sex', ['male', 'female'])
        age = st.slider('Age', 0, 100, 30)
    with col2:
        sibsp = st.slider('Siblings/Spouses Aboard', 0, 8, 0)
        parch = st.slider('Parents/Children Aboard', 0, 6, 0)
        fare = st.slider('Fare Price', 0, 600, 30)
    embarked = st.selectbox('Embarkation Port', ['S', 'C', 'Q'])
    
    # Preprocessing (must match training)
    sex_encoded = 0 if sex == 'male' else 1
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    embarked_encoded = embarked_map[embarked]
    
    # Create input array in CORRECT ORDER
    input_data = np.array([[
        pclass,
        sex_encoded,
        age,
        sibsp,
        parch,
        fare,
        embarked_encoded
    ]])
    
    # Prediction button
    if st.button('Predict Survival', type='primary'):
        try:
            # Debug info
            st.write("Input features:", ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
            st.write("Input values:", input_data[0])
            
            # Make prediction
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)
            
            # Display results
            st.subheader('Prediction Result')
            if prediction[0] == 1:
                st.success('✅ This passenger would have survived!')
            else:
                st.error('❌ This passenger would not have survived.')
            
            # Show probability gauge
            prob_percent = probability[0][1] * 100
            st.metric(label="Survival Probability", 
                     value=f"{prob_percent:.1f}%",
                     delta=f"Confidence: {min(prob_percent, 100-prob_percent):.1f}%")
            
            # Visual probability indicator
            st.progress(int(prob_percent))
            
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {str(e)}")
            st.info("Check that all input values are valid numbers.")

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
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    importance = model.feature_importances_
    
    fig, ax = plt.subplots()
    sns.barplot(x=importance, y=features, ax=ax)
    ax.set_title('Feature Importance')
    st.pyplot(fig)

# Run the app
if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)
