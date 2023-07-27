import streamlit as st
import pandas as pd
import numpy as np
import os
import time

from unet_predict import SEnergyM, unet_path, target_appliances

st.set_page_config(
    page_title="SEnergyM",
    page_icon="🧊",
    layout="wide",
    # initial_sidebar_state="expanded",
)

hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.title('SEnergy')

UKDALE_DIR = "https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2015/UK-DALE-disaggregated"

@st.cache_data
def load_data():
    data = pd.read_csv("https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2015/UK-DALE-disaggregated/house_1/channel_1.dat", delim_whitespace=True, header=None, names=['index', 'real_power'], nrows=1000)
    data = data.set_index("index")
    return data.values.reshape(-1)

data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

first_seq = data[:100].tolist()

model = SEnergyM(unet_path, target_appliances, first_seq)

len_data = model.get_appliances_power().shape[0]

chart_data  = pd.DataFrame({
    'aggregated': model.get_aggregated_power()[len_data-100:len_data],
    target_appliances[0]: model.get_appliances_power()[len_data-100:len_data, 0],
    target_appliances[1]: model.get_appliances_power()[len_data-100:len_data, 1],
    target_appliances[2]: model.get_appliances_power()[len_data-100:len_data, 2],
    target_appliances[3]: model.get_appliances_power()[len_data-100:len_data, 3],
    target_appliances[4]: model.get_appliances_power()[len_data-100:len_data, 4],
})


chart = st.line_chart(chart_data, height=500)

for ind in range(100, data.shape[0]):
    model.predict(data[ind])
    new_data  = pd.DataFrame({
        'aggregated': np.array([model.get_aggregated_power()[-1]]),
        target_appliances[0]: np.array([model.get_appliances_power()[-1, 0]]),
        target_appliances[1]: np.array([model.get_appliances_power()[-1, 1]]),
        target_appliances[2]: np.array([model.get_appliances_power()[-1, 2]]),
        target_appliances[3]: np.array([model.get_appliances_power()[-1, 3]]),
        target_appliances[4]: np.array([model.get_appliances_power()[-1, 4]]),
    })
    chart.add_rows(new_data)
    time.sleep(1)