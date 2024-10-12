from insightface.app import FaceAnalysis
import numpy as np
import pandas as pd
import cv2

import redis

#insight face

from sklearn.metrics import pairwise
#time
import time
from datetime import datetime
import os
#connect to Redis Client
hostname='redis-18068.c61.us-east-1-3.ec2.redns.redis-cloud.com'
portnumber=18068
password='seHLbQiGpJ7RLSEkKzN8QQ9FxGB4oo9y'

r=redis.StrictRedis(host=hostname,
                    port=portnumber,
                    password=password)

#retrive data from database
def retrieve_data(name):
    retrieve_dict=r.hgetall(name)
    retrieve_series=pd.Series(retrieve_dict)
    retrieve_series=retrieve_series.apply(lambda x:np.frombuffer(x,dtype=np.float32))
    index=retrieve_series.index
    index=list(map(lambda x: x.decode(),index))
    retrieve_series.index=index
    retrieve_df=retrieve_series.to_frame().reset_index()
    retrieve_df.columns=['id_name_country','Facial Features']
    retrieve_df[['ID','Name','Country']]=retrieve_df['id_name_country'].apply(lambda x: x.split('@')).apply(pd.Series)
    mydf=retrieve_df[['ID','Name','Country','Facial Features']].copy()
    return retrieve_df

#configure face analysis
faceapp=FaceAnalysis(name='buffalo_sc',
                     root='insightface_model',
                     providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0,det_size=(640,640),det_thresh=0.5)

#ML Search algorithm
def ml_search_algorithm(dataframe,feature_column,test_vector,id_name_country=['ID','Name','Country'],thresh=0.5):
    # step1: take the dataframe( collection of data0
    dataframe=dataframe.copy()
    
    # step 2: index face embedding
    X_list=dataframe[feature_column].tolist()
    X=np.asarray(X_list)
    
    # step 3: calculate cosine similarity
    similar=pairwise.cosine_similarity(X,test_vector.reshape(1,-1))
    similar_arr=np.array(similar).flatten()
    dataframe['cosine']=similar_arr
    
    # step 4: filter the data
    data_filter=dataframe.query(f'cosine>={thresh}')
    if len(data_filter)>0:
        # step 5: get the person name
        data_filter.reset_index(drop=True,inplace=True)
        argmax=data_filter['cosine'].argmax()
        person_id,person_name,person_country=data_filter.loc[argmax][id_name_country]
    else:
        person_id='unknown'
        person_name='unknown'
        person_country='unknown'
   
    return person_id,person_name,person_country
### Real time Prediction
#we need to save logs in redis database
class RealTimePred:
    def __init__(self):
        self.logs = dict(id=[],name=[],country=[],current_time=[])
        
    def reset_dict(self):
        self.logs = dict(id=[],name=[],country=[],current_time=[])
        
    def saveLogs_redis(self):
        # step-1: create a logs dataframe
        dataframe = pd.DataFrame(self.logs)        
        # step-2: drop the duplicate information (distinct name)
        dataframe.drop_duplicates('id',inplace=True) 
        # step-3: push data to redis database (list)
        # encode the data
        id_list = dataframe['id'].tolist()
        name_list = dataframe['name'].tolist()
        country_list=dataframe['country'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []
        for id,name,country,ctime in zip(id_list,name_list, country_list, ctime_list):
            if id != 'unknown':
                concat_string = f"{id}@{name}@{country}@{ctime}"
                encoded_data.append(concat_string)
                
        if len(encoded_data) >0:
            r.lpush('attendance:logs',*encoded_data)
        
                    
        self.reset_dict()
    def face_prediction(self,test_image,dataframe,feature_column,test_vector,id_name_country=['ID','Name','Country'],thresh=0.5):
        #step 0: find the time
        current_time=str(datetime.now())
        #step 1: take the test image and apply insight face
        results=faceapp.get(test_image)
        test_copy=test_image.copy()
        #step 2: use for loop and extraxt embedding
        for res in results:
            x1,y1,x2,y2=res['bbox'].astype(int)
            embeddings=res['embedding']
            person_id,person_name,person_country=ml_search_algorithm(dataframe, 
                                                    feature_column,
                                                    test_vector=embeddings, 
                                                    id_name_country=id_name_country,
                                                    thresh=thresh)
            if person_name=='unknown':
                color=(0,0,255) #bgr
                text_gen='unknown'
            else:
                color=(0,255,0)
                text_gen=person_id+', '+person_country
            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
            
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1)
            cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1)
            # save info in logs dict
            self.logs['id'].append(person_id)
            self.logs['name'].append(person_name)
            self.logs['country'].append(person_country)
            self.logs['current_time'].append(current_time)
        return test_copy
    

#registration form
class RegistrationForm:
    def __init__(self):
        self.sample=0
    def reset(self):
        self.sample=0
    def get_embedding(self,frame):
        #get results from insightface model
        results=faceapp.get(frame,max_num=1)
        embeddings=None
        for res in results:
            self.sample+=1
            x1,y1,x2,y2=res['bbox'].astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
            #put text samples info
            text=f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            #facial features
            embeddings=res['embedding']
        return frame,embeddings
    def save_data_in_redis_db(self,id,name,country):
        #validation of name,id and country
        if id or name is not None:
            if id.strip() !='' and name.strip()!='':
                key=f'{id}@{name}@{country}'
            else:
                return "id_name_false"
        else:
            return 'id_name_false'
        #if face_embedding.txt exists
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'
        #step1: load 'face_embedding
        x_array=np.loadtxt('face_embedding.txt',dtype=np.float32) #flatten array
        #step2: convert into array (proper shape)
        recieved_samples=int(x_array.size/512)
        x_array=x_array.reshape(recieved_samples,512)
        x_array=np.asarray(x_array)
        #step3: calculate mean embeddings)
        x_mean=x_array.mean(axis=0)
        x_mean=x_mean.astype(np.float32)
        x_mean_bytes=x_mean.tobytes()


        #step4: save into redis database
        #redis hashes
        r.hset(name='kdu_students:register',key=key,value=x_mean_bytes)
        #
        os.remove('face_embedding.txt')
        self.reset()

        return True