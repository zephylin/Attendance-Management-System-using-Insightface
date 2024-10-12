import streamlit as st


st.set_page_config(page_title='Attendance System', layout='wide')

st.header('Attendance system using face recognition')

with st.spinner("Loading Models and Connecting to Redis database..."):
    import face_rec

st.success('Model loaded successfully')
st.success('Redis database successfully connected')
