import streamlit as st

st.set_page_config(page_title='Reporting',layout='wide')
from Home import face_rec
from datetime import datetime,date
st.subheader('Reporting')
#retrieve logs and show in this file
#extract data from redis list
name='attendance:logs'
def load_logs(name,end=-1):
    logs_list=face_rec.r.lrange(name,start=0, end=end) 
    #extract all data from redis database (as we used end of -1)
    return logs_list
#let me try to retrieve data for specific time
import pandas as pd
new_logs=[]
x=date.today()
if x==date.today():
    df=pd.DataFrame(columns=['ID','Name','Arrival Time'])
def load_logs_time(name,end=-1):
    logs_list=face_rec.r.lrange(name,start=0, end=end) 
    #my_dict=({'ID':['student ID'],'Name':['student name'],'Arrival Time':['time of arrival']})
    #df=pd.DataFrame(my_dict)

   
    id_list=[]
    name_list=[]
    time1_list=[]
    # for i in logs_list:
    #     i=i.decode()
    #     time=i.split('@')[3]
    #     id=i.split('@')[0]
    #     name=i.split('@')[1]
    #     time1=time
    #     time=time.split()[0]
        
    #     time=datetime.strptime(time,'%Y-%m-%d').date()
    #     if (time==date.today()):
    #         id_list.append(id)
    #         name_list.append(name)
    #         time1_list.append(time1)
    #         if (id in df['ID'].values):
    #             continue
    #         else:
    #             new_row=pd.Series({'ID':id,'Name':name,'Arrival Time':time1})
    #             df.loc[len(df)]=new_row
    #         #row=({'ID':[id_list],'Name':[name_list],'Arrival Time':[time1]})
    #         #new_df=pd.DataFrame(row)
    #         #my_df=pd.concat([df,new_df],ignore_index=True)
            
            
    # return df

    for x in range(len(logs_list)-1,0,-1):
        i=logs_list[x]
        i=i.decode()
        time=i.split('@')[3]
        id=i.split('@')[0]
        name=i.split('@')[1]
        time1=time
        time=time.split()[0]
        
        time=datetime.strptime(time,'%Y-%m-%d').date()
        if (time==date.today()):
            # id_list.append(id)
            # name_list.append(name)
            # time1_list.append(time1)
        
            if (id in df['ID'].values):
                continue
            else:
                new_row=pd.Series({'ID':id,'Name':name,'Arrival Time':time1})
                df.loc[len(df)]=new_row
            #row=({'ID':[id_list],'Name':[name_list],'Arrival Time':[time1]})
            #new_df=pd.DataFrame(row)
            #my_df=pd.concat([df,new_df],ignore_index=True)
            
            
    return df
#tabs to show info
tab2,tab1,tab3= st.tabs(['Registered Data','Logs','Attendance sheet'])
with tab1:
    if st.button('Refresh Logs'):
        st.write(load_logs(name=name))
with tab2:
    if st.button('Refresh data'):
        with st.spinner('Retrieving data from Redis database...'):

            redis_face_db=face_rec.retrieve_data(name='kdu_students:register')
            st.table(redis_face_db[['ID','Name','Country']])
with tab3:
    if st.button("View today's attendance sheet"):
        attendance_df=load_logs_time(name=name)
        st.table(attendance_df)
