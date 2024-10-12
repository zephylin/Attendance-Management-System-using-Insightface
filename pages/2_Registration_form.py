from Home import st
from Home import face_rec
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av

st.set_page_config(page_title='Registration Form',layout='centered')

st.subheader('Registration Form')
#init 
registration_form=face_rec.RegistrationForm()
#step 1: collect person ID, name and country
#form
person_id=st.text_input(label='Student ID',placeholder='Student ID')
person_name=st.text_input(label='Student Name',placeholder='First & Last Name')
person_country=st.selectbox(label='Select Your Country',options=('Bangladesh','Burundi',
                                                         'India','Khazakstan',
                                                         'Kenya','Liberia',
                                                         'Nepal','Rwanda',
                                                         'South Sudan','Tanzania',
                                                         'Uganda','Uzbeskistan'))
                                                        
                                                        
#step2: collect facial embedding of that person
def video_callback_func(frame):
    img=frame.to_ndarray(format='bgr24') # 3d array, bgr
    reg_img,embedding=registration_form.get_embedding(img)
    #two step process
    #step1: save datra into local computer (txt or npz)
    if embedding is not None:
        with open('face_embedding.txt',mode='ab') as f: #ab means appends as byte (read, write and append)
            np.savetxt(f,embedding)
    #step2: 

    return av.VideoFrame.from_ndarray(reg_img,format='bgr24')

webrtc_streamer(key='registration',video_frame_callback=video_callback_func)

#step3: save the data in the redis
if st.button('Submit'):
    return_val=registration_form.save_data_in_redis_db(person_id,person_name,person_country)
    if return_val==True:
        st.success(f"{person_id}  registered successfully")
    elif return_val=='id_name_false':
        st.error('Please enter the name: Name can not be empty or spaces')

    elif return_val=='file_false':
        st.error('Face_embedding.txt not found, Please refresh and execute page again')
