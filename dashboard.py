import pandas as pd
import numpy as np
import scipy
import math
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

import pickle

st.title('Flight Delay Prediction Web App')
st.text('Hello world! This is flight prediction app by The Ourliers :)')

dep_time = st.text_input(label='Please input departure time in format HHMM: (ex: 1230 for 12:30)')
st.write('Departure date: ', dep_time)

distance = st.text_input(label='Please input travel distance in km:')
st.write('Departure distance: ', distance)
year = 2020

day_of_week = st.text_input(label='Please input day you will depart in format ddmmyyyy: (ex: 20052020)')
day = int(day_of_week[:2])
month = int(day_of_week[2:4])
year = int(day_of_week[4:])
st.write('Your departure date: ', day, month, year)
date = datetime.datetime(year, month, day)
day_of_week = date.today().weekday() + 1
st.write('Your departure day of the week: ', day_of_week)


origin = st.text_input(label='Please input your airport origin: (ex: OGD)')
st.write('Your origin is: ', origin)

# new_input = []
# new_input.append(int(dep_time))

new_input = np.array([[int(dep_time), int(distance), year, 0.000e+00, 0.000e+00, 1.000e+00,
       0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
       0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
       0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
       0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00]])


# Change day of week
if day_of_week == 1:
    to_append = [1, 0, 0, 0, 0, 0, 0]
elif day_of_week == 2:
    to_append = [0, 1, 0, 0, 0, 0, 0]
elif day_of_week == 3:
    to_append = [0, 0, 1, 0, 0, 0, 0]
elif day_of_week == 4:
    to_append = [0, 0, 0, 1, 0, 0, 0]
elif day_of_week == 5:
    to_append = [0, 0, 0, 0, 1, 0, 0]
elif day_of_week == 6:
    to_append = [0, 0, 0, 0, 0, 1, 0]
else:
    to_append = [0, 0, 0, 0, 0, 0, 1]

new_input[0][3:10] = to_append

# change origin
day_append = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
if origin == 'SHR':
    day_append[0] = 1
elif origin == 'OGD':
    day_append[1] = 1
elif origin == 'PSM':
    day_append[2] = 1
elif origin == 'HGR':
    day_append[3] = 1
elif origin == 'ASE':
    day_append[4] = 1
elif origin == 'OGS':
    day_append[5] = 1
elif origin == 'HTS':
    day_append[6] = 1
elif origin == 'PAE':
    day_append[7] = 1
elif origin == 'ART':
    day_append[8] = 1
elif origin == 'JLN':
    day_append[9] = 1
elif origin == 'DAB':
    day_append[10] = 1
elif origin == 'ABY':
    day_append[11] = 1
elif origin == 'EKO':
    day_append[12] = 1
elif origin == 'LBF':
    day_append[13] = 1
elif origin == 'OME':
    day_append[14] = 1
elif origin == 'ITO':
    day_append[15] = 1
elif origin == 'PIR':
    day_append[16] = 1
elif origin == 'BTM':
    day_append[17] = 1
elif origin == 'LWS':
    day_append[18] = 1
elif origin == 'CPR':
    day_append[19] = 1


filename = 'stage1_mode.sav'
loaded_model = pickle.load(open(filename, 'rb'))

result = loaded_model.predict(new_input)
is_delayed = 'Your flight will be delayed' if result == 1 else "Congrats!\nYour flight won't be delayed"

st.subheader('Flight prediction:')
st.write(is_delayed)

