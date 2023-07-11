import streamlit as st
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import firebase_admin
from firebase_admin import credentials
from firebase_admin import auth

# Check if Firebase Admin SDK has been initialized
if not firebase_admin._apps:
    # Initialize Firebase Admin SDK
    cred = credentials.Certificate('firebase_credentials.json')
    firebase_admin.initialize_app(cred)

# Load the Tatoeba dataset
data = pd.read_csv("file.csv")

# Preprocess the data
x = np.array(data["Text"])
y = np.array(data["language"])

# Vectorize the text data
cv = CountVectorizer()
X = cv.fit_transform(x)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)

# Train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Set page title
st.title('Language Detection Tool')

# Initialize SessionState
if 'signed_in' not in st.session_state:
    st.session_state.signed_in = False

if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'

# Global user variable
user = None

# Function to display sidebar and sign-in/sign-up options
def display_sidebar():
    st.sidebar.subheader('Welocome')
    option = st.sidebar.selectbox('Go to:', ['Sign Up', 'Sign In'])

    return option

# Function to perform sign-up
def perform_signup():
    email = st.sidebar.text_input('Email')
    password = st.sidebar.text_input('Password', type='password')

    if st.sidebar.button('Sign Up'):
        try:
            global user
            user = auth.create_user(
                email=email,
                password=password
            )
            st.success('Successfully signed up!')

        except ValueError as e:
            st.error('Error signing up: ' + str(e))

# Function to perform sign-in
def perform_signin():
    email = st.sidebar.text_input('Email')
    password = st.sidebar.text_input('Password', type='password')

    if st.sidebar.button('Sign In'):
        try:
            global user
            user = auth.get_user_by_email(email)
            st.sidebar.success('Successfully signed in!')
            st.session_state.signed_in = True

        except ValueError as e:
            st.error('Error signing in: ' + str(e))

# Function to perform sign-out
def perform_signout():
    global user
    if user:
        auth.revoke_refresh_tokens(user.uid)
    user = None
    st.session_state.signed_in = False
    st.sidebar.success('Successfully signed out!')

def det_lang():
    st.write('Enter some text and the tool will predict the language of the text.')
    user_input = st.text_area('Enter a Text', height=200)
    temp = st.button('Detect Language')
    if user_input:
        data = cv.transform([user_input]).toarray()
        output = model.predict(data)
        if temp:
            st.write('Detected Language : ', output[0])

# Display sign-up or sign-in option
if not st.session_state.signed_in:
    option = display_sidebar()

    # Perform sign-up
    if option == 'Sign Up':
        perform_signup()

    # Perform sign-in
    elif option == 'Sign In':
        perform_signin()

if st.session_state.signed_in:
    st.sidebar.button('Sign Out', on_click=perform_signout)

if st.session_state.signed_in:
    det_lang()



# Check if user is signed in
if not st.session_state.signed_in:
    st.warning('Please sign in to use the language detection tool.')
