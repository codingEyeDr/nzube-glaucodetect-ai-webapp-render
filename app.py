# 📦 Import libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
import hashlib
import groq
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# ⚙️ Load environment variables
load_dotenv()

# ⚙️ Streamlit config
st.set_page_config(page_title="NzubeGlaucoDetect AI", layout="wide")

# 🛠 Initialize PostgreSQL for Render
def get_db_connection():
    return psycopg2.connect(os.environ['DATABASE_URL'])

def init_db():
    conn = get_db_connection()
    with conn.cursor() as c:
        c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE,
            fullname TEXT,
            password TEXT
        )''')
        c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            username TEXT,
            patient_name TEXT,
            age INTEGER,
            iop REAL,
            country TEXT,
            eye TEXT,
            result TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
    conn.commit()
    conn.close()

# 🔒 Auth functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, fullname, password):
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute(
                'INSERT INTO users (username, fullname, password) VALUES (%s, %s, %s)',
                (username, fullname, hash_password(password))
            )
        conn.commit()
        return True
    except psycopg2.IntegrityError:
        return False
    finally:
        conn.close()

def get_user(username):
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as c:
        c.execute('SELECT * FROM users WHERE username = %s', (username,))
        return c.fetchone()
    conn.close()

def save_prediction(username, patient_name, age, iop, country, eye, result):
    conn = get_db_connection()
    with conn.cursor() as c:
        c.execute('''
            INSERT INTO predictions (username, patient_name, age, iop, country, eye, result)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (username, patient_name, age, iop, country, eye, result))
    conn.commit()
    conn.close()

def get_user_predictions(username):
    conn = get_db_connection()
    with conn.cursor(cursor_factory=RealDictCursor) as c:
        c.execute('''
            SELECT patient_name, age, iop, country, eye, result, timestamp
            FROM predictions WHERE username=%s
            ORDER BY timestamp DESC
        ''', (username,))
        return c.fetchall()
    conn.close()

# 🤖 Load AI model
# 🤖 Load AI model
@st.cache_resource
def load_glaucoma_model():
    # First check if file exists
    if not os.path.exists("static/NzubeGlaucoma_AI_Predictor.h5"):
        st.error("⚠️ Model file not found at: static/NzubeGlaucoma_AI_Predictor.h5")
        st.warning("Please ensure: \n1. The file exists \n2. It's in the static folder")
        return None
    
    try:
        # Explicitly specify custom objects if needed
        return load_model("static/NzubeGlaucoma_AI_Predictor.h5", 
                        compile=False,
                        custom_objects=None)
    except Exception as e:
        st.error(f"⚠️ Model loading failed. Technical details: {str(e)}")
        st.warning("Possible causes: \n1. TensorFlow version mismatch \n2. Corrupted model file")
        return None

model = load_glaucoma_model()
if model is None:
    st.warning("⚠️ AI model failed to load. Prediction features disabled.")

# 🔍 Prediction function
def predict_glaucoma(image, model):
    if model is None:
        return "Model not loaded - please contact admin"
    
    try:
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Add error handling for prediction
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])
        
        if confidence > 0.5:
            return f"Glaucoma Detected ({confidence:.2%} confidence)"
        else:
            return f"No Glaucoma Detected ({1-confidence:.2%} confidence)"
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Prediction failed"

# 💬 Chatbot function
def query_groq_chatbot(prompt):
    try:
        client = groq.Client(api_key=os.environ['GROQ_API_KEY'])
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "system", "content": "You are an expert ophthalmologist AI. Provide accurate, concise information about glaucoma and eye health."}
            ]
        
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        full_response = ""
        message_placeholder = st.empty()
        
        for chunk in client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=st.session_state.chat_history,
            temperature=0.3,
            max_tokens=512,
            stream=True
        ):
            full_response += (chunk.choices[0].delta.content or "")
            message_placeholder.markdown(full_response + "▌")
            
        message_placeholder.markdown(full_response)
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
        return "Our medical chatbot is temporarily unavailable. Please try again later."

# 📦 Session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_fullname' not in st.session_state:
    st.session_state.user_fullname = ''
if 'db_initialized' not in st.session_state:
    init_db()
    st.session_state.db_initialized = True

# 🔐 Login
def login():
    st.title("🔐 Login to NzubeGlaucoDetect AI")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        user = get_user(username)
        if user and hash_password(password) == user['password']:
            st.session_state.logged_in = True
            st.session_state.user_fullname = user['fullname']
            st.success(f"Welcome, {user['fullname']}! 🎉")
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
        st.title("👁️ NzubeGlaucoDetect AI by Dr. Anthony")
        st.markdown("<hr style='border:1px solid #ddd'>", unsafe_allow_html=True)
        st.subheader("Your AI-powered assistant for glaucoma screening")
        st.markdown("""
        ✅ Upload fundus images separately for **right and left eyes**  
        ✅ Get instant AI predictions  
        ✅ View your prediction history  
        ✅ Chat with our integrated AI chatbot  

        ⚠️ *Disclaimer*: This tool is for educational purposes only. If you have an eye problem, see an Eye Doctor (Optometrist/Ophthalmologist) for professional diagnosis and treatment.
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
                    if model is not None:  # <- Add this check
                        result_right = predict_glaucoma(img_right, model)
                        st.success(f"Prediction: **{result_right}**")
                        save_prediction(st.session_state.user_fullname, patient_name, age, iop_right, country, "Right", result_right)
                    else:
                        st.error("Model not available for predictions")  # <- Add this fallback

        with col2:
            st.subheader("👁️ Left Eye")
            iop_left = st.number_input("Intraocular Pressure (Left Eye, mmHg)", min_value=0.0, step=0.1)
            left_eye_image = st.file_uploader("Upload Left Eye Fundus Image", type=["jpg", "jpeg", "png"], key="left")
            if left_eye_image:
                img_left = Image.open(left_eye_image)
                st.image(img_left, caption='Left Eye Fundus', use_container_width=True)
                if st.button("Predict Left Eye"):
                    if model is not None:  # <- Add this check
                        result_left = predict_glaucoma(img_left, model)
                        st.success(f"Prediction: **{result_left}**")
                        save_prediction(st.session_state.user_fullname, patient_name, age, iop_left, country, "Left", result_left)
                    else:
                        st.error("Model not available for predictions")  # <- Add this fallback

    elif page == "📊 History":
        st.title("📊 Prediction History")
        st.markdown("---")
        history = get_user_predictions(st.session_state.user_fullname)
        if history:
            for record in history:
                st.markdown(f"""
                ✅ **Date:** {record['timestamp']} | **Patient:** *{record['patient_name']}*  
                👁️ **Eye:** {record['eye']} | **Result:** {record['result']}  
                ℹ️ **Age:** {record['age']} | **IOP:** {record['iop']} mmHg | **Country:** {record['country']}
                """)
                st.markdown("---")
        else:
            st.info("No predictions found yet.")

    elif page == "💬 Chatbot":
        st.title("💬 AI Ophthalmology Assistant")
        st.caption("Ask me anything about glaucoma or eye health")
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        for message in st.session_state.chat_history:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        if prompt := st.chat_input("Type your question..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                query_groq_chatbot(prompt)

    elif page == "ℹ️ About":
        st.title("ℹ️ About NzubeGlaucoDetect AI")
        st.markdown("""
        NzubeGlaucoDetect AI helps screen for glaucoma by analyzing fundus images with deep learning.

        **Created by:** Dr. Anthony Anyanwu  
        **Built with:** Streamlit, TensorFlow, Groq AI
        """)

    elif page == "🚪 Logout":
        st.session_state.logged_in = False
        st.session_state.user_fullname = ''
        st.session_state.chat_history = []
        st.success("You have been logged out")