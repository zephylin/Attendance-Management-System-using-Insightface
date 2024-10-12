import streamlit as st
from streamlit_webrtc import webrtc_streamer
from Home import face_rec
# st.set_page_config(page_title='Registration form', layout='wide')
st.subheader('Real-Time Attendance System')

import av
import time




#retrieve data from redis
with st.spinner('Retrieving data from Redis database...'):

    redis_face_db=face_rec.retrieve_data(name='kdu_students:register')
    #st.dataframe(redis_face_db)
    st.success('Data successfully retrieved from Redis database')
st.write('Now you can verify your face by pressing START button')
st.write('Note: No Second camera installed, Please face the laptop web cam.')
# time 
waitTime = 30 # time in sec
setTime = time.time()
realtimepred = face_rec.RealTimePred() # real time prediction class
#streamlit webrtc


#call back function
def video_frame_callback(frame):
    global setTime

    img = frame.to_ndarray(format="bgr24")# bgr24: 3D numpy array
    #operation u can perform on an array
    pred_img=realtimepred.face_prediction(img,redis_face_db,
                                      'Facial Features',
                                      ['ID','Name','Country'],
                                      thresh=0.5)

    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime:
        realtimepred.saveLogs_redis()
        setTime = time.time() # reset time        
        print('Save Data to redis database')

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="realtimeprediction", video_frame_callback=video_frame_callback)