import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import graph_objs as go
import statsmodels.api as sm
from fbprophet import Prophet
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

INPUT_DATAFRAME = os.environ.get('INPUT_DATAFRAME')

class InformationClass():

    def __init__(self):
        """
        This class give some brief information about the dataset
        """
        print("Information object created")

    def _get_missing_values(self, data):
        """
        Find missing dates of given sample
        """
        
        missing_values = pd.date_range(start = data.index.min(), end = data.index.max() ).difference(data.index)
        
        return missing_values

    def info(self, data):
        """
        print various info about dataset
        """
        self.missing_values=self._get_missing_values(data)

        print("=" * 50)

        print(f'Data range from {data.index.min().strftime("%d.%m.%Y")} to {data.index.max().strftime("%d.%m.%Y")}')
        print(f'Total number of samples: {len(data)}')
        if len(self.missing_values) > 0:
            print(f'Total number of of missing dates from timeseries {len(self.missing_values)}:')
            print(', '.join(self.missing_values.strftime("%d.%m.%Y").tolist()))
            print('\'backfill\' by default will be user for resample')
        print("="*50)

class DataProcessClass():

    def __init__(self):
        """
        Preprocess class for dataset
        """
        print("PreprocessClass object created")

    def preprocess(self, data, strategy_type):
        return data.resample('D').fillna(strategy_type)

    def convert_for_prophet(self, data):
        df = data.reset_index()
        df.columns = ['ds', 'y']
        # converting timezones (issue https://github.com/facebook/prophet/issues/831)
        df['ds'] = df['ds'].dt.tz_localize(None)
        return df

    def split_data(self, data, period):
        return data[:-period]

    def make_comparison_dataframe(self, historical, forecast):
        """
        Join the history with the forecast.
        The resulting dataset will contain columns 'yhat', 'yhat_lower', 'yhat_upper' and 'y'.
        """
        return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))
    
class VisualizerClass:
    
    def __init__(self):
        print("Visualizer object created!")
    
    def raw_data_plot(self, df, title):
        
        plt.figure(figsize=(15, 12))
        plt.plot(df.index,df['value'], 'r-', label = 'value')
        plt.title('Database trends')
        plt.ylabel('value)');
        plt.legend();
        plt.show()

        # common_kw = dict(x=df.index, mode='lines')
        # data = [go.Scatter(y=df[c], name=c, **common_kw) for c in df.columns]
        # layout = dict(title=title)
        # fig = dict(data=data, layout=layout)
        # iplot(fig, show_link=False)
        st.pyplot(plt)

    def moving_average_plot(self, actual_df, series, n_days):

        """
        series - dataframe with timeseries
        n - rolling window size 

        """
        n = n_days * 24

        rolling_mean = series.rolling(window=n).mean()

        rolling_std =  series.rolling(window=n).std()
        upper_bond = rolling_mean+1.96*rolling_std
        lower_bond = rolling_mean-1.96*rolling_std

        plt.figure(figsize=(15,5))
        plt.title("Moving average\n window size = {}".format(n))
        plt.plot(rolling_mean, "g", label="Rolling mean trend")

        plt.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        plt.plot(lower_bond, "r--")
        plt.plot(actual_df[n:], label="Actual values")
        plt.legend(loc="upper left")
        plt.grid(True)
        st.pyplot(plt)


    def decomposition_plot(self, data):
        
        decomposition = sm.tsa.seasonal_decompose(data, model='additive')
        fig = decomposition.plot()
        fig.set_size_inches(15, 8)
        plt.show()
        st.pyplot(plt)

    def show_forecast(self, cmp_df, num_predictions, num_values, title):
        """Visualize the forecast."""

        def create_go(name, column, num, **kwargs):
            points = cmp_df.tail(num)
            args = dict(name=name, x=points.index, y=points[column], mode='lines')
            args.update(kwargs)
            return go.Scatter(**args)

        lower_bound = create_go('Lower Bound', 'yhat_lower', num_predictions,
                                line=dict(width=0),
                                marker=dict(color="gray"))
        upper_bound = create_go('Upper Bound', 'yhat_upper', num_predictions,
                                line=dict(width=0),
                                marker=dict(color="gray"),
                                fillcolor='rgba(68, 68, 68, 0.3)', 
                                fill='tonexty')
        forecast = create_go('Forecast', 'yhat', num_predictions,
                             line=dict(color='rgb(31, 119, 180)'))
        actual = create_go('Actual', 'y', num_values,
                           marker=dict(color="red"))

        # In this case the order of the series is important because of the filling
        data = [lower_bound, upper_bound, forecast, actual]

        layout = go.Layout(yaxis=dict(title='Posts'), title=title, showlegend = False)
        fig = go.Figure(data=data, layout=layout)
        iplot(fig, show_link=False)
        st.plotly_chart(fig)


class ProphetClass():

    def __init__(self):

        print("ProphetClass Created")

    def fit_predict_forecast(self, data, period):
        
        self.clf = Prophet(daily_seasonality=True)
        self.clf.fit(data)
        future = self.clf.make_future_dataframe(periods=period)
        return self.clf.predict(future)

    def calculate_forecast_errors(self, df, period):
        """Calculate MAPE and MAE of the forecast.

           Args:
               df: joined dataset with 'y' and 'yhat' columns.
               prediction_size: number of days at the end to predict.
        """

        # Make a copy
        df = df.copy()

        # Now we calculate the values of e_i and p_i according to the formulas given in the article above.
        df['e'] = df['y'] - df['yhat']
        df['p'] = 100 * df['e'] / df['y']

        # Recall that we held out the values of the last `prediction_size` days
        # in order to predict them and measure the quality of the model. 

        # Now cut out the part of the data which we made our prediction for.
        predicted_part = df[-period:]

        # Define the function that averages absolute error values over the predicted part.
        error_mean = lambda error_name: np.mean(np.abs(predicted_part[error_name]))

        # Now we can calculate MAPE and MAE and return the resulting dictionary of errors.
        return {'MAPE': error_mean('p'), 'MAE': error_mean('e')}
        
        
class SampleClass():

    def __init__(self, input_database, input_metric):
        """

        :param sample: data will be used for modelling and evaluation
        """
        data = pd.read_csv(INPUT_DATAFRAME)
        data['rollup_timestamp'] = pd.to_datetime(data['rollup_timestamp'])
        data = data.set_index('rollup_timestamp')

        self.sample = data[data['database'] == input_database][[input_metric]]
        self.sample.rename(columns={input_metric: "value"}, inplace=True)
        self.sample.index = pd.to_datetime(self.sample.index)

        print("SampleClass object created")

        self.number_of_data=self.sample.shape[0]
        
        #Create instance of objects
        self.info = InformationClass()
        self.dataprocess = DataProcessClass()
        self.visualizer = VisualizerClass()
        self.prophet = ProphetClass()

    def _get_df(self):
        return self.sample

    def _get_series(self):
        return self.sample['value']

    def _get_index_freq(self):
        return self.sample.index.freq

    def _get_shape(self):
        return self.sample.shape
    
    def _number_of_data(self):
        return self.sample.shape[0]
    
    def information(self):
        """
        using _info object gives summary about dataset
        :return:
        """
        self.info.info(self.sample)

    def preprocess_data(self, strategy_type="backfill"):
        """
        Process data depend upon strategy type
        :param strategy_type: Preprocessing strategy type
        :return:
        """
        self.strategy_type=strategy_type
        
        self.sample = self.dataprocess.preprocess(self.sample, strategy_type)
        self.prophet_ds = self.dataprocess.convert_for_prophet(self.sample)

    def raw_data_plot(self):
        self.visualizer.raw_data_plot(self.sample, 'Database metric value (daily)')

    def moving_average_plot(self, n):
        self.visualizer.moving_average_plot(self.sample, self._get_series(), n)

    def decomposition_plot(self):
        self.visualizer.decomposition_plot(self.sample)

    def split_data_by_period(self, period):
        self.period_len = abs(period)
        if period < 0:
            self.period_past, self.period_future = self.period_len, 0
            self.train_df = self.dataprocess.split_data(self.prophet_ds, self.period_past)
        else:
            self.period_past, self.period_future = 0, self.period_len
            self.train_df = self.prophet_ds

    def machine_learning(self):
        self.forecast = self.prophet.fit_predict_forecast(self.train_df, self.period_future)

    def make_compasion(self):
        self.cmp_df = self.dataprocess.make_comparison_dataframe(self.prophet_ds, self.forecast)

    def print_forecast_errors(self):
        for err_name, err_value in self.prophet.calculate_forecast_errors(self.cmp_df, self.period_past).items():
            print(err_name, err_value)

    def show_forecast(self):
        self.visualizer.show_forecast(self.cmp_df, self.period_len, self.period_len*3, 'Database metric value forecast')