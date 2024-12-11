import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from math import ceil

#load data
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=['date'])

#file paths
file_paths = {
    "Weekly": "data/ts_weekly.csv",
    "Monthly": "data/ts_monthly.csv"
}

#mapping of variable names to human readable style
ts_names = {
    'ryaay' : 'RYAAY price',
    'dln_ryaay' : 'Log return RYAAY',
    'avg_rating' : 'Avg. TP rating',
    'avg_sentiment' : 'Avg. TP sentiment',
    'guests' : 'No. of guests (mn)',
    'load_factor' : 'Load factor (ratio)',
    'd_avg_rating' : 'FD of avg. TP rating',
    'd_avg_sentiment' : 'FD of avg. TP sentiment',
    'd_guests' : 'FD of no. of guests (mn)',
    'd_load_factor' : 'FD of load factor (ratio)',
    'sd_guests' : 'Seasonal diff. of guests (mn)',
}

model_names = {
    'rating_sent_only_logpred' : 'Rating & sentiment only ',
    'time_dummies_logpred' : '~ with time dummies ',
    'guests_qinteraction_logpred' : '~ with guests x quarter ',
    'sd_guests_model_logpred' : '~ with SD of guests ',
    'guests_loadf_qinteraction_logpred' : '~ with guests & load fact. x quarter ',
    'sd_guests_loadf_qinteraction_logpred' : '~ with SD of guests and load fact. x quarter ',
    'rating_sent_only_levelpred' : 'Rating & sentiment only',
    'time_dummies_levelpred' : '~ with time dummies',
    'guests_qinteraction_levelpred' : '~ with guests x quarter',
    'sd_guests_model_levelpred' : '~ with SD of guests',
    'guests_loadf_qinteraction_levelpred' : '~ with guests & load fact. x quarter',
    'sd_guests_loadf_qinteraction_levelpred' : '~ with SD of guests and load fact. x quarter',
    'weekly_rating_sent_only_logpred' : 'Rating & sentiment only ',
    'weekly_time_dummies_logpred' : '~ with time dummies ',
    'weekly_rating_sent_only_levelpred' : 'Rating & sentiment only',
    'weekly_time_dummies_levelpred' : '~ with time dummies',
}

#sidebar UI
st.sidebar.header("DASHBOARD SETTINGS")
data_type = st.sidebar.radio("Select data type:", ["Monthly", "Weekly"])
view_type = st.sidebar.radio("Select view:", ["Time series", "Regression results"])

#load the data that has been selected
df = load_data(file_paths[data_type])

#exclude some variables from plotting
excluded_vars = ["war", "lockdown", "quarter", "year", "week", "month"]
excluded_vars += [col for col in df.columns if "logpred" in col or "levelpred" in col]

#let's configure the time series plots
#these are basically the recreations of the EDA plots
if view_type == "Time series":
    st.title(f"{data_type} time series")
    #the user can select which vars it wants to plot
    selectable_vars = [col for col in df.columns if col not in excluded_vars + ['date']]
    selected_vars = st.multiselect("Select variables to plot:", selectable_vars,
                                   default = ['ryaay', 'dln_ryaay'],
                                   format_func = lambda x: ts_names[x])

    #I plot each var on a separate chart in a grid layout
    if selected_vars:
        num_charts = len(selected_vars)
        cols = 2
        rows = ceil(num_charts / cols)

        grid_figures = []
        for var in selected_vars:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df[var], mode='lines', name=var))
            fig.update_layout(
                title=ts_names[var],
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_white",
                height=300
            )
            grid_figures.append(fig)

        for i in range(rows):
            cols_to_display = grid_figures[i * cols:(i + 1) * cols]
            col_count = len(cols_to_display)
            col_layout = st.columns(col_count)
            for col, chart in zip(col_layout, cols_to_display):
                with col:
                    st.plotly_chart(chart, use_container_width=True)
    else:
        st.warning("Select at least one variable to plot.")

#let's also configure the regression results plots
elif view_type == "Regression results":
    st.title(f"{data_type} regression results")
    
    #the user can select the prediction type
    prediction_type = st.radio("Select prediction type:", ["Level price", "Log returns"])
    relevant_models = [
        col for col in df.columns 
        if ("logpred" in col and prediction_type == "Log returns") or 
           ("levelpred" in col and prediction_type == "Level price")
    ]
    
    #and also the model specifications
    def_modspec = ''
    if data_type == 'Monthly':
        if prediction_type == 'Level price':
            def_modspec = 'rating_sent_only_levelpred'
        else:
            def_modspec = 'rating_sent_only_logpred'
    else:
        if prediction_type == 'Level price':
            def_modspec = 'weekly_rating_sent_only_levelpred'
        else:
            def_modspec = 'weekly_rating_sent_only_logpred'

    selected_models = st.multiselect("Select model specifications:", relevant_models,
                                     default = def_modspec,
                                     format_func = lambda x: model_names[x])

    #I also plot always either the true level price or the true log returns
    if selected_models:
        target_var = "dln_ryaay" if prediction_type == "Log returns" else "ryaay"
        fig = go.Figure()
        for model in selected_models:
            fig.add_trace(go.Scatter(x=df['date'], y=df[model], mode='lines', name=model_names[model]))

        fig.add_trace(go.Scatter(x=df['date'], y=df[target_var], mode='lines', name=ts_names[target_var]))

        fig.update_layout(
            title=f"{prediction_type} predictions",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Select at least one model to plot.")
