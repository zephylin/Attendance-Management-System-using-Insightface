import streamlit as st


st.set_page_config(page_title='Attendance System', layout='wide')
with st.spinner("Loading Models and Connecting to Redis database..."):
    import face_rec
st.header('Attendance system using face recognition')



st.success('Model loaded successfully')
st.success('Redis database successfully connected')
