import os
import streamlit as st
import pandas as pd
import numpy as np
import trends

INPUT_DATAFRAME = os.environ.get('INPUT_DATAFRAME')

st.title("Database trends for Oracle DBA")
st.markdown(
"""
This is a small demo of a database stats forecasting based on Oracle Cloud Control metrics
""")

@st.cache(persist=True)

def load_data():
    data = pd.read_csv(INPUT_DATAFRAME)
    data = data.set_index('rollup_timestamp')
    return data

data = load_data()

metrics_dict = {'size_gd': 'Database size in Gb per day', 
               'dbtime': 'DB time per day', 
               'redosize_mb': 'Redo size in Mb per day'}

input_database = st.sidebar.selectbox(
     'Select database id for analysis',
     data.database.unique())

input_metric = st.sidebar.selectbox('Select metric for analysis', list(metrics_dict.keys()), format_func=lambda x: metrics_dict[x])

sample = trends.SampleClass(input_database, input_metric)
sample.preprocess_data()

st.write(f"You selected {input_database} database")
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(sample._get_df())

st.write(f"Please select buttons from left sidebar for various tools!")

if st.sidebar.button("Graph for raw data"):
    st.title("Graph for raw data:")
    sample.raw_data_plot()
days = st.sidebar.slider('Rolling window size in days for averaging', 1, 10, 1)
if st.sidebar.button("Graph with moving avetage"):
    st.title("Graph with moving avetage:")
    sample.moving_average_plot(days)
if st.sidebar.button("Graph for decomposition"):
    st.title("Graph for decomposition:")
    sample.decomposition_plot()
period = st.sidebar.slider('Prediction period', -30, 90, 30)
if st.sidebar.button("Build model and predict"):
    st.title("Graph for actual data and model prediction:")
    sample.split_data_by_period(period)
    sample.machine_learning()
    sample.make_compasion()
    if st.checkbox('Show prediction data'):
        st.subheader('Raw data')
        st.write(sample.forecast)
    sample.show_forecast()
    if period < 0:
        st.write('Calculated errors for this model:')
        for err_name, err_value in sample.prophet.calculate_forecast_errors(sample.cmp_df, sample.period_past).items():
            st.write(err_name, err_value)
