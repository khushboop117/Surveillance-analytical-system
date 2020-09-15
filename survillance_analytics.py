import cv2
import pandas as pd
import csv
import numpy as np
import pafy
import imutils
import time
from past.builtins import xrange
from imutils import convenience
from imutils.convenience import is_cv3
from PIL import Image
import PIL.ImageDraw
from collections import Counter
from matplotlib import pyplot as plt
import dlib
import streamlit as st
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
import io
import urllib
import webbrowser
import base64
import progressbar
from io import BytesIO
import os
import json
import pickle
import uuid
import re
from tqdm import tqdm_gui

import threading 
from multiprocessing import Process
from datetime import datetime
from bokeh.plotting import figure

def progress_bar(dur):
    my_bar = st.progress(0)
    for percent_complete in range(dur):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)

def get_cap(location):
    print("Loading in function", str(location))
    video_stream = cv2.VideoCapture(str(location))

    if (video_stream.isOpened() == False):
        print("Error opening video  file")
    return video_stream


def plot_1(up, down,gend, agen):
    df = pd.DataFrame(data={"age": agen, "gender": gend})
    df.to_csv("agen.csv", sep=',',index=False)
    df1 = pd.DataFrame(data={"Up": up, "Down": down})
    df1 = df1.drop_duplicates()
    df1.to_csv("count.csv", sep=',',index=False)
    pd.read_csv("agen.csv")
    plt.plot(up,label = "up counter")
    plt.plot(down,label = "down counter")
    plt.xlabel("frame")
    plt.ylabel("count")
    plt.legend()
    st.pyplot()
    df.groupby(['age','gender']).size().unstack().plot(kind='bar',stacked=True)
    st.pyplot()


def download_button(object_to_download, download_filename, button_text, pickle_it=False):
        if pickle_it:
            try:
                object_to_download = pickle.dumps(object_to_download)
            except pickle.PicklingError as e:
                st.write(e)
                return None

        else:
            if isinstance(object_to_download, bytes):
                pass

            elif isinstance(object_to_download, pd.DataFrame):
                object_to_download = object_to_download.to_csv(index=False)

                            # Try JSON encode for everything else
            else:
                object_to_download = json.dumps(object_to_download)

        try:
            # some strings <-> bytes conversions necessary here
            b64 = base64.b64encode(object_to_download.encode()).decode()

        except AttributeError as e:
            b64 = base64.b64encode(object_to_download).decode()

        button_uuid = str(uuid.uuid4()).replace('-', '')
        button_id = re.sub('\d+', '', button_uuid)

        custom_css = f""" 
            <style>
                #{button_id} {{
                    background-color: rgb(255, 255, 255);
                    color: rgb(38, 39, 48);
                    padding: 0.25em 0.38em;
                    position: relative;
                    text-decoration: none;
                    border-radius: 4px;
                    border-width: 1px;
                    border-style: solid;
                    border-color: rgb(230, 234, 241);
                    border-image: initial;
                }} 
                #{button_id}:hover {{
                    border-color: rgb(246, 51, 102);
                    color: rgb(246, 51, 102);
                }}
                #{button_id}:active {{
                    box-shadow: none;
                    background-color: rgb(246, 51, 102);
                    color: white;
                    }}
            </style> """

        dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'

        return dl_link

def  age_gender(vs,dur):
    totalFrames = 0
    totalDown = 0
    totalUp = 0
    rects=[]
    agen = []
    gend = []
    writer = None
    W= None
    H= None
    up = []
    down = []
    skipFrames = rate
    status = ""
    net = cv2.dnn.readNet("deploy.prototxt", "deploy.caffemodel")
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    font = cv2.FONT_HERSHEY_SIMPLEX
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
    gender_list = ['Male', 'Female']
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
    frameST = st.empty()
    trackers = []
    trackableObjects = {}


    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)

    widgets = ['Loading: ', progressbar.AnimatedMarker()]
    bar = progressbar.ProgressBar(widgets=widgets).start() 
    
    with st.spinner('Processing...'):
        while True:
            frame = vs.read()
            frame = frame[1]

            if frame is None:
                break

            frame = imutils.resize(frame, width=320)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if W is None or H is None:
                (H, W) = frame.shape[:2]

            status = "Waiting"
            
            rects = []
            
            if totalFrames % skipFrames < skipFrames/5:
                status = "Detecting"
                trackers = []

                face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                for (x, y, w, h )in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

                    face_img = frame[y:y+h, h:h+w].copy()
                    blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

                    gender_net.setInput(blob)
                    gender_preds = gender_net.forward()
                    gender = gender_list[gender_preds[0].argmax()]
                    print("Gender : " + gender)
                    age_net.setInput(blob)
                    age_preds = age_net.forward()
                    age = age_list[age_preds[0].argmax()]
                    print("Age Range: " + age)
                    overlay_text = "%s %s" % (gender, age)
                    cv2.putText(frame, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    agen.append(age)
                    gend.append(gender)
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

                net.setInput(blob)
                detections = net.forward()

                for i in np.arange(0, detections.shape[2]):

                    confidence = detections[0, 0, i, 2]
                    if confidence > confidence_threshold:
                        idx = int(detections[0, 0, i, 1])

                        if CLASSES[idx] not in ("person"):
                            continue
         

                        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
          
                        (startX, startY, endX, endY) = box.astype("int")
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(rgb, rect)
                        trackers.append(tracker)
            else:
              for tracker in trackers:
                status = "Tracking"

                tracker.update(rgb)
                pos = tracker.get_position()

                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                rects.append((startX, startY, endX, endY))

            cv2.line(frame, (0, 0), (W, 0), (0, 255, 255), 2)

            objects = ct.update(rects)

            for (objectID, centroid) in objects.items():
                to = trackableObjects.get(objectID, None)


                if to is None:
                    to = TrackableObject(objectID, centroid)


                else:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)

                    if not to.counted:
                        if direction < 0 and centroid[1] < H // 2:
                            totalUp += 1
                            to.counted = True


                        elif direction > 0 and centroid[1] > H // 2:
                            totalDown += 1
                            to.counted = True

                up.append(totalUp)
                down.append(totalDown)
                trackableObjects[objectID] = to


                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
               
            info = [("Up", totalUp),("Down", totalDown),("Status", status),]


            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            totalFrames += 1
            frameST.image(frame, channels="BGR",caption='output video', use_column_width=True)
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter("output.avi", fourcc, rate,(frame.shape[1], frame.shape[0]), True)

        return up, down, gend, agen



 

page = st.sidebar.radio("Navigate to: ",('Homepage','App','Older Statistics'))
if page == "Homepage":
    image = Image.open('lg.jpeg')
    st.image(image,use_column_width=True)
    st.title("Surveillance Analytics System ")
    st.write("Nowadays every building or residential area is equipped with surveillance system like CCTV cameras, which monitors it’s associated area and generates data in form of video recording of around 2 to 3 gigabytes per day. Despite some professional building which has dedicated team to monitor the CCTV footage, no one in this busy life style has time to view these recordings. And even with the dedicated monitoring team there are chances that they might miss the important details as humans are error prone. To tackle this task we are presenting a machine learning based novel approach to generate analytics from the footage. This system will be able to detect persons with their gender along with their predictive age group. It will keep count of how many people are entering / leaving the building along with timestamp.")
    st.write('Whether it is a shopping center, retail chain, museum, library, sporting venue, bank, restaurant or building, there always needs to be guard to maintain the log of people entering the building. It is the one of the accurate and reliable ways to measure people’s activity in and out of physical space. One can also get critical insights around visitor age, gender in real-time with our intelligent person counter which uses advanced machine learning concepts.')
    st.header("Applications")
    st.write("Every business needs to measure the customer traffic in order to make useful decisions of the business. Knowing the age and gender of the person will help them to know the demographics of their customers and hence will allow them to make useful decisions for their welfare.")
    st.write("In the retail industry, this device will help them to count the number of customers entering and existing a retail store.  Knowing the people counter will help them to optimize the need of staff based on the customer. ")
    st.write("Libraries and museums often receive funding from various organizations whose value depends on the people visiting it hence this people counter will help the organizations to make decisions.")
    st.write("Airports are one of the busiest and most used places in the world. From security check to boarding the flights, all places are heavily crowded. So this tool will help to manage the crews to allow efficient operations.")
    st.write("Shopping Malls draw lots of visitor traffic. With people counting, you will be able to see which areas are most crowded  so that one can plan accordingly.")
    st.header("Use Cases")
    st.write("- Retail store / Shopping mall where lot of people enter / leave throughout the day can benefit from this system, by getting daily count of how many people comes for shopping with gender identification and predicted age group. Once the system is installed and data is generated for at least a year, one can integrate forecasting system for monthly or weekly prediction on people rush. Based on the gender and age group of people comes in store at particular time, store owner can arrange items that group of people requires most.")
    st.write("- Restaurants can benefit from this system by identifying which age group of people comes, based on that they can adjust theme of the restaurants on can add food items in the menu. For example if data tells that children comes many times then they can add food item that children likes most and can create some fun zone where children can play so their parents can eat peacefully. Another group is old age people; if that is most frequent age group then restaurant can add item such is suitable to eat for old age people. ")
    st.header("Features")   
    st.write("- Real time Analysis from cctv feed and File upload option to get analysis on particular video.")
    st.write("- Additional option to get Analysis directly from YouTube video link.")
    st.write("- State of the art deep learning models for person detection and counting, Gender detection and Age group prediction.")
    st.write("- Interactive dashboard with lot of visualization to showcase the analytical aspects.")
    st.write("- Download option to save the processed system view video.")
    st.write("- Add on to export frame wise analysis in csv format.")
    st.balloons()

elif page == "App":

    st.title("Surveillance Analytics System ")
    confidence_threshold = st.sidebar.slider("Confidence threshold" ,3.0,0.0, 1.0)
    app_mode = st.sidebar.selectbox("Select a source: ",[ "Youtube URL", "Device"])
    if app_mode == "Youtube URL":
        user_input = st.text_input("Enter url below: ")
        if st.button("Enter"): 
            url = user_input
            
            video = pafy.new(url)
            duration = video.duration

            ftr = [3600,60,1]

            dur = sum([a*b for a,b in zip(ftr, map(int,duration.split(':')))])
            vid_best = video.getbest(preftype="mp4")     
            vs = cv2.VideoCapture()
            vs.open(vid_best.url)
            
            rate = vs.get(cv2.CAP_PROP_FPS)
            est_tot_frames = dur
            print("Estimated time to process",est_tot_frames)
            up, down, gend, agen= age_gender(vs,dur)
            st.success('done!')


            st.title("plots")
            plot_1(up,down, gend, agen)

            with open('age1.csv', 'rb') as f:
                s = f.read()
            download_button_str = download_button(s, './age1.csv', 'Age CSV file')
            st.sidebar.markdown(download_button_str, unsafe_allow_html=True)

            with open('./count.csv', 'rb') as f:
                s = f.read()
            download_button_str1 = download_button(s,'./count.csv', 'count')
            st.sidebar.markdown(download_button_str1, unsafe_allow_html=True)

            with open('./output.avi', 'rb') as f:
                s = f.read()
            download_button_str1 = download_button(s,'output.avi', 'output file')
            st.sidebar.markdown(download_button_str1, unsafe_allow_html=True)


    elif app_mode == "Device":
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4","avi"])
        temporary_location = False

        if uploaded_file is not None:
            g = io.BytesIO(uploaded_file.read())  
            temporary_location = "testout_simple.mp4"

            with open(temporary_location, 'wb') as out: 
                out.write(g.read())

            vs = get_cap(temporary_location)
            rate = vs.get(cv2.CAP_PROP_FPS)
            dur = 49
            up1, down1, gend1, agen1= age_gender(vs,dur)
            st.success('done!')
            st.title("plots")
            plot_1(up1,down1,gend1,agen1)

            with open("agen.csv", 'rb') as f:
                s = f.read()
            download_button_str = download_button(s, './age1.csv', 'Agen CSV file')
            st.sidebar.markdown(download_button_str, unsafe_allow_html=True)

            with open('./count.csv', 'rb') as f:
                s = f.read()
            download_button_str1 = download_button(s,'./count.csv', 'count CSV file')
            st.sidebar.markdown(download_button_str1, unsafe_allow_html=True)
            with open('./output.avi', 'rb') as f:
                s = f.read()
            download_button_str1 = download_button(s,'output.avi', 'output file')
            st.sidebar.markdown(download_button_str1, unsafe_allow_html=True)

if page == "Older Statistics":
    day_data = st.sidebar.selectbox("Select a day: ",[ "Last 7 days","Day 1","Day 2","Day 3","Day 4","Day 5","Day 6","Day 7"])
    if day_data == "Last 7 days":
        st.title("PLOTS")
        st.header("Up and Down counter Plot")
        df0 = pd.read_csv("count_final.csv")
        st.line_chart(df0)
        st.header("Age and gender plot")
        df = pd.read_csv("final_age.csv")
        df.groupby(['age','gender']).size().unstack().plot(kind='bar',stacked=True)
        st.pyplot()
    if day_data == "Day 1":
        st.title("PLOTS")
        st.header("Up and Down plot for Day 1")
        df1 = pd.read_csv("d1_count.csv")
        st.line_chart(df1)
        st.header("Age and gender plot for Day1")
        df = pd.read_csv("d1_age.csv")
        df.groupby(['age','gender']).size().unstack().plot(kind='bar',stacked=True)
        st.pyplot()
    if day_data == "Day 2":
        st.title("PLOTS")
        st.header("Up and Down plot for Day 2")
        df2 = pd.read_csv("d2_count.csv")
        st.line_chart(df2)
        st.header("Age and gender plot for Day 2")
        df = pd.read_csv("d2_age.csv")
        df.groupby(['age','gender']).size().unstack().plot(kind='bar',stacked=True)
        st.pyplot()
    if day_data == "Day 3":
        st.title("PLOTS")
        st.header("Up and Down plot for Day 3")
        df3 = pd.read_csv("d3_count.csv")
        st.line_chart(df3)
        st.header("Age and gender plot for Day 3")
        df = pd.read_csv("d3_age.csv")
        df.groupby(['age','gender']).size().unstack().plot(kind='bar',stacked=True)
        st.pyplot()
    if day_data == "Day 4":
        st.header("Up and Down plot for Day 4")
        df4 = pd.read_csv("d4_count.csv")
        st.line_chart(df4)
        st.header("Age and gender plot for Day 4")
        df = pd.read_csv("d4_age.csv")
        df.groupby(['age','gender']).size().unstack().plot(kind='bar',stacked=True)
        st.pyplot()
    if day_data == "Day 5":
        st.title("PLOTS")
        st.header("Up and Down plot for Day 5")
        df5 = pd.read_csv("d5_count.csv")
        st.line_chart(df5)
        st.header("Age and gender plot for Day 5")
        df = pd.read_csv("d5_age.csv")
        df.groupby(['age','gender']).size().unstack().plot(kind='bar',stacked=True)
        st.pyplot()
    if day_data == "Day 6":
        st.title("PLOTS")
        st.header("Up and Down plot for Day 6")
        df6 = pd.read_csv("d6_count.csv")
        st.line_chart(df6)
        st.header("Age and gender plot for Day 6")
        df = pd.read_csv("d6_age.csv")
        df.groupby(['age','gender']).size().unstack().plot(kind='bar',stacked=True)
        st.pyplot()
    if day_data == "Day 7":
        st.title("PLOTS")
        st.header("Up and Down plot for Day 7")
        df7 = pd.read_csv("d7_count.csv")
        st.line_chart(df7)
        st.header("Age and gender plot for Day 7")
        df = pd.read_csv("d7_age.csv")
        df.groupby(['age','gender']).size().unstack().plot(kind='bar',stacked=True)
        st.pyplot()
