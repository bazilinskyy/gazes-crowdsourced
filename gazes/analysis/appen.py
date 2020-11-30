# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>

import json
import os
import pandas as pd
from tqdm import tqdm

import gazes as gz

logger = gz.CustomLogger(__name__)  # use custom logger


class Appen:
    file_data = []  # list of files with appen data
    appen_data = pd.DataFrame()  # pandas dataframe with extracted data
    save_p = False  # save data as pickle file
    load_p = False  # load data as pickle file
    save_csv = False  # save data as csv file
    file_p = 'appen_data.p'  # pickle file for saving data
    file_csv = 'appen_data.csv'  # csv file for saving data

    def __init__(self,
                 file_data: list,
                 save_p: bool,
                 load_p: bool,
                 save_csv: bool):
        self.file_data = file_data
        self.save_p = save_p
        self.load_p = load_p
        self.save_csv = save_csv

    def read_data(self):
        # load data
        if self.load_p:
            self.appen_data = gz.common.load_from_p(self.file_p,
                                                    'appen data')
        # process data
        else:
            # load from csv
            self.appen_data = pd.read_csv(self.file_data)
            # set index to worker code
            self.appen_data = self.appen_data.set_index('type_the_code_that_you_received_at_the_end_of_the_experiment')
        # save to pickle
        if self.save_p:
            gz.common.save_to_p(self.file_p,  self.appen_data, 'appen data')
        # save to
        if self.save_csv:
            self.appen_data.to_csv(gz.settings.output_dir + '/' +
                                   self.file_csv)
            logger.info('Saved appen data to csv file {}.', self.file_csv)

        print(self.appen_data.head)
        print(self.appen_data.keys())

        # return df with data
        return self.appen_data
