import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = (
    "input/database_data.csv"
)

st.title("Database trends for Oracle DBA")
st.markdown(
"""
This is a small demo of a database stats forecasting based on Oracle Cloud Control metrics
""")

@st.cache(persist=True)

def load_data():
    data = pd.read_csv(DATA_PATH)
    data = data.set_index('rollup_timestamp')
    return data

data = load_data()

metrics_dict = {'redosize_mb': 'Redo size in Mb per day', 
               'dbtime': 'DB time per day', 
               'size_gd': 'Database size in Gb per day'}

input_database = st.sidebar.selectbox(
     'Select database id for analysis',
     data.database.unique())

input_metric = st.sidebar.selectbox('Select metric for analysis', list(metrics_dict.keys()), format_func=lambda x: metrics_dict[x])


st.write(f"You selected {input_database} database")

sample = data[data['database'] == input_database][[input_metric]]
sample.rename(columns={input_metric: "value"}, inplace=True)
sample.index = pd.to_datetime(sample.index)
sample = sample.resample('D').fillna("backfill")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(sample)

st.title("Graphs")
st.write("Graph for raw data")
plt.figure(figsize=(15, 12))
plt.plot(sample.index, sample['value'], 'r-', label = 'value')
plt.title('Database trends')
plt.ylabel('value)');
plt.legend();
st.pyplot()

if st.sidebar.button('Click Func foo'):

    import statsmodels.api as sm
    from pylab import rcParams
    st.write("Graph for trend, seasonal and noise components")

    rcParams['figure.figsize'] = 18, 8
    decomposition = sm.tsa.seasonal_decompose(sample, model='additive')
    fig = decomposition.plot()
    st.pyplot(fig)