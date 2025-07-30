# 📦 Import libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import sqlite3
import hashlib
import os
from groq import Groq

# ⚙️ Streamlit page config
st.set_page_config(page_title="GlaucoDetect AI", layout="wide")

# 🛠 Initialize SQLite database
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

# Create tables if they don't exist
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    fullname TEXT,
    password TEXT
)
''')
c.execute('''
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    patient_name TEXT,
    age INTEGER,
    iop REAL,
    country TEXT,
    eye TEXT,
    result TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')
conn.commit()

# 🔒 Helper functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(hashed_password, user_password):
    return hashed_password == hashlib.sha256(user_password.encode()).hexdigest()

def add_user(username, fullname, password):
    try:
        c.execute('INSERT INTO users (username, fullname, password) VALUES (?, ?, ?)',
                  (username, fullname, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def get_user(username):
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    return c.fetchone()

def save_prediction(username, patient_name, age, iop, country, eye, result):
    c.execute('''
        INSERT INTO predictions (username, patient_name, age, iop, country, eye, result)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (username, patient_name, age, iop, country, eye, result))
    conn.commit()

def get_user_predictions(username):
    c.execute('''
        SELECT patient_name, age, iop, country, eye, result, timestamp
        FROM predictions WHERE username=?
        ORDER BY timestamp DESC
    ''', (username,))
    return c.fetchall()

# 🤖 Load AI model
@st.cache_resource
def load_glaucoma_model():
    return load_model("NzubeGlaucoma_AI_Predictor.h5")

model = load_glaucoma_model()

# 🔍 Prediction function
def predict_glaucoma(image, model):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Glaucoma Detected" if prediction[0][0] > 0.5 else "No Glaucoma Detected"

# 🆕 Enhanced Groq Chatbot
def query_groq_chatbot(prompt):
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        
        # Initialize chat history with medical context
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "system", "content": "You are an expert ophthalmologist AI. Provide accurate, concise information about glaucoma and eye health."}
            ]
        
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Stream the response
        full_response = ""
        message_placeholder = st.empty()
        
        for chunk in client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=st.session_state.chat_history,
            temperature=0.3,  # Lower for medical accuracy
            max_tokens=512,
            stream=True
        ):
            full_response += (chunk.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
            
        # Update history and display final response
        message_placeholder.markdown(full_response)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        
    except Exception as e:
        st.error(f"⚠️ System Error: {str(e)}")
        return "Our medical chatbot is temporarily unavailable. Please try again later."

# 📦 Session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_fullname' not in st.session_state:
    st.session_state.user_fullname = ''

# 🔐 Login
def login():
    st.title("🔐 Login to GlaucoDetect AI")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        user = get_user(username)
        if user and check_password(user[3], password):
            st.session_state.logged_in = True
            st.session_state.user_fullname = user[2]
            st.success(f"Welcome, {user[2]}! 🎉")
        else:
            st.error("Invalid username or password")

# 📝 Signup
def signup():
    st.title("📝 Sign Up")
    fullname = st.text_input("Full Name")
    username = st.text_input("Choose a Username")
    password = st.text_input("Password", type='password')
    confirm = st.text_input("Confirm Password", type='password')
    if st.button("Register"):
        if password != confirm:
            st.error("Passwords do not match")
        elif get_user(username):
            st.error("Username already exists")
        else:
            if add_user(username, fullname, password):
                st.success("Registration successful! You can now log in.")
            else:
                st.error("Something went wrong. Try again.")

# 🚀 Main app
if not st.session_state.logged_in:
    menu = st.sidebar.radio("Menu", ["Login", "Sign Up"])
    if menu == "Login":
        login()
    else:
        signup()
else:
    st.sidebar.image("logo.png", width=150)
    st.sidebar.title(f"👤 {st.session_state.user_fullname}")
    page = st.sidebar.radio("📋 Menu", ["🏠 Home", "🔍 Predict", "📊 History", "💬 Chatbot", "ℹ️ About", "🚪 Logout"])

    if page == "🏠 Home":
        st.title("👁️ GlaucoDetect AI by Dr. Anthony")
        st.markdown("<hr style='border:1px solid #ddd'>", unsafe_allow_html=True)
        st.subheader("Your AI-powered assistant for glaucoma screening")
        st.markdown("""
        ✅ Upload fundus images separately for **right and left eyes**  
        ✅ Get instant AI predictions  
        ✅ View your prediction history  
        ✅ Chat with our integrated AI chatbot  

        ⚠️ *Disclaimer*: This tool is for educational purposes only.
        """)

    elif page == "🔍 Predict":
        st.title("🔍 Fundus Image Prediction")
        st.markdown("---")
        st.markdown("### 🧑‍⚕️ Patient Information")
        patient_name = st.text_input("Patient Full Name")
        age = st.number_input("Age", min_value=1, max_value=120, step=1)
        country = st.text_input("Country")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("👁️ Right Eye")
            iop_right = st.number_input("Intraocular Pressure (Right Eye, mmHg)", min_value=0.0, step=0.1)
            right_eye_image = st.file_uploader("Upload Right Eye Fundus Image", type=["jpg", "jpeg", "png"], key="right")
            if right_eye_image:
                img_right = Image.open(right_eye_image)
                st.image(img_right, caption='Right Eye Fundus', use_container_width=True)
                if st.button("Predict Right Eye"):
                    result_right = predict_glaucoma(img_right, model)
                    st.success(f"Prediction: **{result_right}**")
                    save_prediction(st.session_state.user_fullname, patient_name, age, iop_right, country, "Right", result_right)

        with col2:
            st.subheader("👁️ Left Eye")
            iop_left = st.number_input("Intraocular Pressure (Left Eye, mmHg)", min_value=0.0, step=0.1)
            left_eye_image = st.file_uploader("Upload Left Eye Fundus Image", type=["jpg", "jpeg", "png"], key="left")
            if left_eye_image:
                img_left = Image.open(left_eye_image)
                st.image(img_left, caption='Left Eye Fundus', use_container_width=True)
                if st.button("Predict Left Eye"):
                    result_left = predict_glaucoma(img_left, model)
                    st.success(f"Prediction: **{result_left}**")
                    save_prediction(st.session_state.user_fullname, patient_name, age, iop_left, country, "Left", result_left)

    elif page == "📊 History":
        st.title("📊 Prediction History")
        st.markdown("---")
        history = get_user_predictions(st.session_state.user_fullname)
        if history:
            for record in history:
                patient_name, age, iop, country, eye, result, timestamp = record
                st.markdown(f"""
                ✅ **Date:** {timestamp} | **Patient:** *{patient_name}* | **Eye:** {eye} | **Result:** {result}  
                ℹ️ Age: {age} | IOP: {iop} | Country: {country}
                """)
                st.markdown("---")
        else:
            st.info("No predictions found yet.")

    elif page == "💬 Chatbot":
        st.title("💬 AI Ophthalmology Assistant")
        st.caption("Ask me anything about glaucoma or eye health")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat messages (excluding system prompt)
        for message in st.session_state.chat_history:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # User input
        if prompt := st.chat_input("Type your question..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                query_groq_chatbot(prompt)

    elif page == "ℹ️ About":
        st.title("ℹ️ About GlaucoDetect AI")
        st.markdown("""
        GlaucoDetect AI helps screen for glaucoma by analyzing fundus images with deep learning.

        **Created by:** Dr. Anthony  
        **Built with:** Streamlit, TensorFlow, Groq AI
        """)

    elif page == "🚪 Logout":
        st.session_state.logged_in = False
        st.session_state.user_fullname = ''
        st.session_state.chat_history = []
        st.success("You have been logged out")