import streamlit as st


st.set_page_config(page_title='Attendance System', layout='wide')
with st.spinner("Loading Models and Connecting to Redis database..."):
    import face_rec
st.header('Attendance system using face recognition')

st.success('Model loaded successfully')

if face_rec.redis_connected:
    st.success('Redis database successfully connected')
else:
    st.error(f'Redis connection failed: {face_rec.redis_error_msg}')
    st.info('The app will load, but features requiring the database (attendance, registration, reports) will not work. '
            'Please check your `.env` file and ensure your Redis instance is active.')
