import pandas as pd
import numpy as np

class Information():

    def __init__(self):
        """
        This class give some brief information about the datasets.
        """
        print("Information object created")

    def _get_missing_values(self,data):
        """
        Find missing dates of given sample
        """
        #Getting sum of missing values for each feature
        missing_values = pd.date_range(start = data.index.min(), end = data.index.max() ).difference(data.index)
        
        #Returning missing values
        return missing_values

    def info(self,data):
        """
        print various info about dataset
        """
        self.missing_values=self._get_missing_values(data)

        print("=" * 50)

        print(data.min())
        print(self.missing_values)
        print(data.max())

        print("="*50)

class ObjectOrientedSample():

    def __init__(self, path):
        """

        :param sample: data will be used for modelling and evaluation
        """
        self.sample = pd.read_csv(path)
        self.sample['rollup_timestamp'] = pd.to_datetime(self.sample['rollup_timestamp'])
        self.sample = self.sample.set_index('rollup_timestamp')

        print("ObjectOrientedSample object created")

        self.number_of_data=self.sample.shape[0]


        #Create instance of objects
        self._info=Information()


    def _get_all_data(self):
        return pd.concat([self.sample])

    def information(self):
        """
        using _info object gives summary about dataset
        :return:
        """
        self._info.info(self.sample)