# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import json
import os
import pandas as pd
from tqdm import tqdm
import re

import gazes as gz

logger = gz.CustomLogger(__name__)  # use custom logger


class Heroku:
    files_data = []  # list of files with heroku data
    heroku_data = pd.DataFrame()  # pandas dataframe with extracted data
    save_p = False  # save data as pickle file
    load_p = False  # load data as pickle file
    save_csv = False  # save data as csv file
    file_p = 'heroku_data.p'  # pickle file for saving data
    file_csv = 'heroku_data.csv'  # csv file for saving data
    # keys with meta information
    meta_keys = ['worker_code',
                 'browser_user_agent',
                 'browser_app_name',
                 'browser_major_version',
                 'browser_full_version',
                 'browser_name',
                 'group_choice'
                 ]
    # prefixes used for files in node.js implementation
    prefixes = {'training': 'training_',
                'stimulus': 'image_',
                'codeblock': 'cb_',
                'sentinel': 'sentinel_',
                'sentinel_cb': 'sentinel_cb_'
                }

    def __init__(self,
                 files_data: list,
                 save_p: bool,
                 load_p: bool,
                 save_csv: bool):
        self.files_data = files_data
        self.save_p = save_p
        self.load_p = load_p
        self.save_csv = save_csv

    def read_data(self):
        # load data
        if self.load_p:
            self.heroku_data = gz.common.load_from_p(self.file_p,
                                                     'heroku data')
        # process data
        else:
            # read files with heroku data one by one
            data_list = []
            data_dict = {}  # dictionary with data
            for file in self.files_data:
                logger.info('Reading heroku data from {}.', file)
                f = open(file, 'r')
                # add data from the file to the dictionary
                data_list += f.readlines()
                f.close()
            # read rows in data
            for row in tqdm(data_list):  # tqdm adds progress bar
                # use dict to store data
                dict_row = {}
                # load data from a single row into a list
                list_row = json.loads(row)
                # flag that training image was detected
                train_found = False
                # last found training image
                train_name = ''
                # flag that stimulus was detected
                stim_found = False
                # last found stimulus
                stim_name = ''
                # flag that sentinel image was detected
                sent_found = False
                # last found sentinel image
                sent_name = ''
                # go over cells in the row with data
                for data_cell in list_row['data']:
                    # extract meta info form the call
                    for key in self.meta_keys:
                        if key in data_cell.keys():
                            # piece of meta data found, update dictionary
                            dict_row[key] = data_cell[key]
                            if key == 'worker_code':
                                logger.debug('Working with row for worker_code {}.',
                                             data_cell['worker_code'])
                    # check if stimulus data is present
                    if 'stimulus' in data_cell.keys():
                        # extract name of stimulus after last slash
                        stim_no_path = data_cell['stimulus'].rsplit('/', 1)[-1]
                        # remove extension
                        stim_no_path = os.path.splitext(stim_no_path)[0]
                        # Check if it is a block with stimulus and not an
                        # instructions block
                        if gz.common.search_dict(self.prefixes, stim_no_path) != None:
                            # stimulus is found
                            logger.debug('Found stimulus {}.', stim_no_path)
                            if self.prefixes['training'] in stim_no_path:
                                # record that stimulus was detected for the cells
                                # to follow
                                train_found = True
                                train_name = stim_no_path
                            # training image is found
                            elif self.prefixes['stimulus'] in stim_no_path:
                                # record that stimulus was detected for the cells
                                # to follow
                                stim_found = True
                                stim_name = stim_no_path
                            # codeblock for sentinel image is found
                            elif self.prefixes['sentinel_cb'] in stim_no_path:
                                # record codeblock name for last stimulus
                                if sent_name != '':
                                    # extract ID of codeblock
                                    num_found = re.findall(r'\d+', stim_no_path)
                                    # recored ID of codeblock
                                    dict_row[sent_name + '-cb'] = num_found[0]
                            # codeblock image is found
                            elif self.prefixes['codeblock'] in stim_no_path:
                                # record codeblock name for last stimulus or
                                # training image
                                if train_name != '':
                                    # extract ID of codeblock
                                    num_found = re.findall(r'\d+', stim_no_path)
                                    # recored ID of codeblock
                                    dict_row[train_name + '-cb'] = num_found[0]
                                elif stim_name != '':
                                    # extract ID of codeblock
                                    num_found = re.findall(r'\d+', stim_no_path)
                                    # recored ID of codeblock
                                    dict_row[stim_name + '-cb'] = num_found[0]
                            # sentinel image is found
                            elif self.prefixes['sentinel'] in stim_no_path:
                                # record that stimulus was detected for the cells
                                # to follow
                                sent_found = True
                                sent_name = stim_no_path
                    # data entry following a codechart found
                    elif 'responses' in data_cell.keys():
                        # record given input
                        responses = json.loads(data_cell['responses'])
                        logger.debug('Found input {}.',
                                     responses['input-codeblock'])
                        if train_name != '':
                            # turn input to upper case
                            str_in = responses['input-codeblock'].upper()
                            # save inputted value
                            dict_row[train_name + '-in'] = str_in
                            # record time spent on stimulus
                            dict_row[train_name + '-rt'] = data_cell['rt']
                            # reset flags for found stimulus
                            train_found = False
                            train_name = ''
                        if stim_name != '':
                            # turn input to upper case
                            str_in = responses['input-codeblock'].upper()
                            # save inputted value
                            dict_row[stim_name + '-in'] = str_in
                            # record time spent on stimulus
                            dict_row[stim_name + '-rt'] = data_cell['rt']
                            # reset flags for found stimulus
                            stim_found = False
                            stim_name = ''
                        elif sent_name != '':
                            # turn input to upper case
                            str_in = responses['input-codeblock'].upper()
                            # save inputted value
                            dict_row[sent_name + '-in'] = str_in
                            # record time spent on stimulus
                            dict_row[sent_name + '-rt'] = data_cell['rt']
                            # reset flags for found sentinel image
                            sent_found = False
                            sent_name = ''
                try:
                    data_dict[dict_row['worker_code']].update(dict_row)
                except Exception as e:
                    data_dict[dict_row['worker_code']] = dict_row
            # turn into panda's dataframe
            self.heroku_data = pd.DataFrame(data_dict)
            self.heroku_data = self.heroku_data.transpose()
        # save to pickle
        if self.save_p:
            gz.common.save_to_p(self.file_p,  self.heroku_data, 'heroku data')
        # save to csv
        if self.save_csv:
            self.heroku_data.to_csv(gz.settings.output_dir + '/' +
                                    self.file_csv)
            logger.info('Saved heroku data to csv file {}.', self.file_csv)

        print(self.heroku_data.head)
        print(self.heroku_data.keys())

        # return df with data
        return self.heroku_data
