import logging
from typing import Union
from insightface.app import FaceAnalysis
import numpy as np
import pandas as pd
import cv2
import redis
from sklearn.metrics import pairwise
import time
from datetime import datetime
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('face_rec')

# Load environment variables from .env file
load_dotenv()

# Connect to Redis Client using environment variables (never hardcode secrets!)
hostname = os.getenv('REDIS_HOST')
portnumber = os.getenv('REDIS_PORT')
password = os.getenv('REDIS_PASSWORD')

# Track connection state so pages can show appropriate UI
r: redis.StrictRedis | None = None
redis_connected: bool = False
redis_error_msg: str = ''

if not all([hostname, portnumber, password]):
    redis_error_msg = 'Missing Redis environment variables. Check your .env file.'
    logger.error(redis_error_msg)
else:
    try:
        portnumber = int(portnumber)
        logger.info('Connecting to Redis database at %s:%s', hostname, portnumber)
        r = redis.StrictRedis(host=hostname,
                              port=portnumber,
                              password=password)
        # Test the connection (StrictRedis is lazy — this forces a real connection)
        r.ping()
        redis_connected = True
        logger.info('Redis connection established successfully')
    except redis.ConnectionError as e:
        redis_error_msg = f'Failed to connect to Redis: {e}'
        logger.error(redis_error_msg)
        r = None
    except ValueError:
        redis_error_msg = f'REDIS_PORT must be a valid integer, got: {os.getenv("REDIS_PORT")}'
        logger.error(redis_error_msg)
        r = None

#retrive data from database
def retrieve_data(name: str) -> pd.DataFrame:
    if not redis_connected or r is None:
        logger.error('Cannot retrieve data: Redis is not connected')
        return pd.DataFrame(columns=['id_name_country','Facial Features','ID','Name','Country'])
    try:
        logger.info('Retrieving data from Redis key: %s', name)
        retrieve_dict=r.hgetall(name)
        if not retrieve_dict:
            logger.warning('No data found in Redis key: %s', name)
            return pd.DataFrame(columns=['id_name_country','Facial Features','ID','Name','Country'])
        retrieve_series=pd.Series(retrieve_dict)
        retrieve_series=retrieve_series.apply(lambda x:np.frombuffer(x,dtype=np.float32))
        index=retrieve_series.index
        index=list(map(lambda x: x.decode(),index))
        retrieve_series.index=index
        retrieve_df=retrieve_series.to_frame().reset_index()
        retrieve_df.columns=['id_name_country','Facial Features']
        retrieve_df[['ID','Name','Country']]=retrieve_df['id_name_country'].apply(lambda x: x.split('@')).apply(pd.Series)
        mydf=retrieve_df[['ID','Name','Country','Facial Features']].copy()
        logger.info('Retrieved %d records from database', len(retrieve_df))
        return retrieve_df
    except redis.RedisError as e:
        logger.error('Redis error while retrieving data: %s', e)
        return pd.DataFrame(columns=['id_name_country','Facial Features','ID','Name','Country'])
    except Exception as e:
        logger.error('Unexpected error in retrieve_data: %s', e)
        return pd.DataFrame(columns=['id_name_country','Facial Features','ID','Name','Country'])

#configure face analysis
faceapp=FaceAnalysis(name='buffalo_sc',
                     root='insightface_model',
                     providers=['CPUExecutionProvider'])
faceapp.prepare(ctx_id=0,det_size=(640,640),det_thresh=0.5)

#ML Search algorithm
def ml_search_algorithm(
    dataframe: pd.DataFrame,
    feature_column: str,
    test_vector: np.ndarray,
    id_name_country: list[str] = ['ID','Name','Country'],
    thresh: float = 0.5
) -> tuple[str, str, str]:
    # step1: take the dataframe (collection of data)
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
    def __init__(self) -> None:
        self.logs: dict[str, list[str]] = dict(id=[],name=[],country=[],current_time=[])
        
    def reset_dict(self) -> None:
        self.logs = dict(id=[],name=[],country=[],current_time=[])
        
    def saveLogs_redis(self) -> None:
        try:
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
                if not redis_connected or r is None:
                    logger.error('Cannot save logs: Redis is not connected')
                    return
                r.lpush('attendance:logs',*encoded_data)
                logger.info('Saved %d attendance log(s) to Redis', len(encoded_data))
        except redis.RedisError as e:
            logger.error('Failed to save attendance logs to Redis: %s', e)
        except Exception as e:
            logger.error('Unexpected error in saveLogs_redis: %s', e)
        finally:
            self.reset_dict()
    def face_prediction(
        self,
        test_image: np.ndarray,
        dataframe: pd.DataFrame,
        feature_column: str,
        id_name_country: list[str] = ['ID','Name','Country'],
        thresh: float = 0.5
    ) -> np.ndarray:
        #step 0: find the time
        current_time=str(datetime.now())
        test_copy=test_image.copy()
        
        try:
            #step 1: take the test image and apply insight face
            results=faceapp.get(test_image)
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
        except Exception as e:
            logger.error('Error during face prediction: %s', e)
            
        return test_copy
    

#registration form
class RegistrationForm:
    def __init__(self) -> None:
        self.sample: int = 0
    def reset(self) -> None:
        self.sample = 0
    def get_embedding(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
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
    def save_data_in_redis_db(self, id: str, name: str, country: str) -> Union[bool, str]:
        #validation of name, id and country
        if id is not None and name is not None:
            if id.strip() !='' and name.strip()!='':
                key=f'{id}@{name}@{country}'
            else:
                return "id_name_false"
        else:
            return 'id_name_false'
        #if face_embedding.txt exists
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'
        
        try:
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
            if not redis_connected or r is None:
                logger.error('Cannot register: Redis is not connected')
                return 'redis_error'
            #redis hashes
            r.hset(name='kdu_students:register',key=key,value=x_mean_bytes)
            logger.info('Registered new user: %s (ID: %s, Country: %s)', name, id, country)
            os.remove('face_embedding.txt')
            self.reset()
            return True
        except redis.RedisError as e:
            logger.error('Redis error during registration: %s', e)
            return 'redis_error'
        except Exception as e:
            logger.error('Unexpected error during registration: %s', e)
            return 'registration_error'