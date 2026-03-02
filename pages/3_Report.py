import streamlit as st
import pandas as pd
from datetime import datetime, date

st.set_page_config(page_title='Reporting', layout='wide')
from Home import face_rec

st.subheader('Reporting')

if not face_rec.redis_connected:
    st.error('Redis is not connected. Reports are unavailable.')
    st.info('Please check your `.env` file and ensure your Redis instance is active, then refresh.')
    st.stop()

LOGS_KEY = 'attendance:logs'


def load_logs(name: str, end: int = -1) -> list[bytes]:
    """Retrieve raw attendance logs from Redis."""
    logs_list = face_rec.r.lrange(name, start=0, end=end)
    return logs_list


def load_todays_attendance(name: str, end: int = -1) -> pd.DataFrame:
    """Parse attendance logs and return today's unique check-ins."""
    logs_list = face_rec.r.lrange(name, start=0, end=end)
    df = pd.DataFrame(columns=['ID', 'Name', 'Arrival Time'])

    for entry in reversed(logs_list):
        decoded = entry.decode()
        parts = decoded.split('@')
        student_id, student_name, arrival_time = parts[0], parts[1], parts[3]
        log_date = datetime.strptime(arrival_time.split()[0], '%Y-%m-%d').date()

        if log_date == date.today() and student_id not in df['ID'].values:
            new_row = pd.Series({
                'ID': student_id,
                'Name': student_name,
                'Arrival Time': arrival_time
            })
            df.loc[len(df)] = new_row

    return df


# Tabs to display information
tab_registered, tab_logs, tab_attendance = st.tabs([
    'Registered Data', 'Logs', 'Attendance Sheet'
])

with tab_logs:
    if st.button('Refresh Logs'):
        st.write(load_logs(name=LOGS_KEY))

with tab_registered:
    if st.button('Refresh Data'):
        with st.spinner('Retrieving data from Redis database...'):
            redis_face_db = face_rec.retrieve_data(name='kdu_students:register')
            st.table(redis_face_db[['ID', 'Name', 'Country']])

with tab_attendance:
    if st.button("View Today's Attendance Sheet"):
        attendance_df = load_todays_attendance(name=LOGS_KEY)
        if attendance_df.empty:
            st.info('No attendance records found for today.')
        else:
            st.table(attendance_df)
