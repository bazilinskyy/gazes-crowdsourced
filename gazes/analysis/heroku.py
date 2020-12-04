# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import gazes as gz

# warning about partial assignment
pd.options.mode.chained_assignment = None  # default='warn'

logger = gz.CustomLogger(__name__)  # use custom logger


class Heroku:
    files_data = []  # list of files with heroku data
    heroku_data = pd.DataFrame()  # pandas dataframe with extracted data
    save_p = False  # save data as pickle file
    load_p = False  # load data as pickle file
    save_csv = False  # save data as csv file
    file_p = 'heroku_data.p'  # pickle file for saving data
    file_data_csv = 'heroku_data.csv'  # csv file for saving data
    file_points_csv = 'points.csv'  # csv file for saving data
    file_points_worker_csv = 'points_worker.csv'  # csv file for saving data
    # keys with meta information
    meta_keys = ['worker_code',
                 'browser_user_agent',
                 'browser_app_name',
                 'browser_major_version',
                 'browser_full_version',
                 'browser_name',
                 'group_choice',
                 'image_ids']
    # prefixes used for files in node.js implementation
    prefixes = {'training': 'training_',
                'stimulus': 'image_',
                'codeblock': 'cb_',
                'sentinel': 'sentinel_',
                'sentinel_cb': 'sentinel_cb_'}

    def __init__(self,
                 files_data: list,
                 save_p: bool,
                 load_p: bool,
                 save_csv: bool):
        self.files_data = files_data
        self.save_p = save_p
        self.load_p = load_p
        self.save_csv = save_csv

    def set_data(self, heroku_data):
        """
        Setter for the data object
        """
        old_shape = self.heroku_data.shape  # store old shape for logging
        self.heroku_data = heroku_data
        logger.info('Updated heroku_data. Old shape: {}. New shape: {}.',
                    old_shape,
                    self.heroku_data.shape)

    def read_data(self):
        # todo: filter data based on mistakes with salient images
        # load data
        if self.load_p:
            df = gz.common.load_from_p(self.file_p,
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
                                logger.debug('{}: working with row with data.',
                                             data_cell['worker_code'])
                    # check if stimulus data is present
                    if 'stimulus' in data_cell.keys():
                        # extract name of stimulus after last slash
                        stim_no_path = data_cell['stimulus'].rsplit('/', 1)[-1]
                        # remove extension
                        stim_no_path = os.path.splitext(stim_no_path)[0]
                        # Check if it is a block with stimulus and not an
                        # instructions block
                        if (gz.common.search_dict(self.prefixes, stim_no_path)
                           is not None):
                            # stimulus is found
                            logger.debug('Found stimulus {}.', stim_no_path)
                            if self.prefixes['training'] in stim_no_path:
                                # Record that stimulus was detected for the
                                # cells to follow
                                train_found = True
                                train_name = stim_no_path
                            # training image is found
                            elif self.prefixes['stimulus'] in stim_no_path:
                                # Record that stimulus was detected for the
                                # cells to follow
                                stim_found = True
                                stim_name = stim_no_path
                            # codeblock for sentinel image is found
                            elif self.prefixes['sentinel_cb'] in stim_no_path:
                                # record codeblock name for last stimulus
                                if sent_name != '':
                                    # extract ID of codeblock
                                    num_found = re.findall(r'\d+',
                                                           stim_no_path)
                                    # Check if codeblocks were recorded previously  # noqa: E501
                                    if sent_name + '-cb' not in dict_row.keys():  # noqa: E501
                                        # first value
                                        dict_row[sent_name + '-cb'] = [num_found[0]]  # noqa: E501
                                    else:
                                        # previous values found
                                        dict_row[sent_name + '-cb'].append(num_found[0])  # noqa: E501
                            # codeblock image is found
                            elif self.prefixes['codeblock'] in stim_no_path:
                                # record codeblock name for last stimulus or
                                # training image
                                if train_name != '':
                                    # extract ID of codeblock
                                    num_found = re.findall(r'\d+',
                                                           stim_no_path)
                                    # Check if codeblocks were recorded previously  # noqa: E501
                                    if train_name + '-cb' not in dict_row.keys():  # noqa: E501
                                        # first value
                                        dict_row[train_name + '-cb'] = [num_found[0]]  # noqa: E501
                                    else:
                                        # previous values found
                                        dict_row[train_name + '-cb'].append(num_found[0])  # noqa: E501
                                elif stim_name != '':
                                    # extract ID of codeblock
                                    num_found = re.findall(r'\d+',
                                                           stim_no_path)
                                    # Check if codeblocks were recorded previously  # noqa: E501
                                    if stim_name + '-cb' not in dict_row.keys():  # noqa: E501
                                        # first value
                                        dict_row[stim_name + '-cb'] = [num_found[0]]  # noqa: E501
                                    else:
                                        # previous values found
                                        dict_row[stim_name + '-cb'].append(num_found[0])  # noqa: E501
                            # sentinel image is found
                            elif self.prefixes['sentinel'] in stim_no_path:
                                # Record that stimulus was detected for the
                                # cells to follow
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
                            # Check if inputted values were recorded previously
                            if train_name + '-in' not in dict_row.keys():
                                # first value
                                dict_row[train_name + '-in'] = [str_in]
                            else:
                                # previous values found
                                dict_row[train_name + '-in'].append(str_in)
                            # Check if time spent values were recorded previously  # noqa: E501
                            if train_name + '-rt' not in dict_row.keys():
                                # first value
                                dict_row[train_name + '-rt'] = [data_cell['rt']]  # noqa: E501
                            else:
                                # previous values found
                                dict_row[train_name + '-rt'].append(data_cell['rt'])  # noqa: E501
                            # reset flags for found stimulus
                            train_found = False
                            train_name = ''
                        if stim_name != '':
                            # turn input to upper case
                            str_in = responses['input-codeblock'].upper()
                            # Check if inputted values were recorded previously  # noqa: E501
                            if stim_name + '-in' not in dict_row.keys():
                                # first value
                                dict_row[stim_name + '-in'] = [str_in]
                            else:
                                # previous values found
                                dict_row[stim_name + '-in'].append(str_in)
                            # Check if time spent values were recorded previously  # noqa: E501
                            if stim_name + '-rt' not in dict_row.keys():
                                # first value
                                dict_row[stim_name + '-rt'] = [data_cell['rt']]
                            else:
                                # previous values found
                                dict_row[stim_name + '-rt'].append(data_cell['rt'])  # noqa: E501
                            # reset flags for found stimulus
                            stim_found = False
                            stim_name = ''
                        elif sent_name != '':
                            # turn input to upper case
                            str_in = responses['input-codeblock'].upper()
                            # Check if inputted values were recorded previously
                            if sent_name + '-in' not in dict_row.keys():
                                # first value
                                dict_row[sent_name + '-in'] = [str_in]
                            else:
                                # previous values found
                                dict_row[sent_name + '-in'].append(str_in)
                            # Check if time spent values were recorded previously  # noqa: E501
                            if sent_name + '-rt' not in dict_row.keys():
                                # first value
                                dict_row[sent_name + '-rt'] = [data_cell['rt']]
                            else:
                                # previous values found
                                dict_row[sent_name + '-rt'].append(data_cell['rt'])  # noqa: E501
                            # reset flags for found sentinel image
                            sent_found = False
                            sent_name = ''
                try:
                    data_dict[dict_row['worker_code']].update(dict_row)
                except Exception as e:
                    data_dict[dict_row['worker_code']] = dict_row
            # turn into panda's dataframe
            df = pd.DataFrame(data_dict)
            df = df.transpose()
            # filter data
            df = self.filter_data(df)
        # save to pickle
        if self.save_p:
            gz.common.save_to_p(self.file_p,  df, 'heroku data')
        # save to csv
        if self.save_csv:
            df.to_csv(gz.settings.output_dir + '/' + self.file_data_csv)
            logger.info('Saved heroku data to csv file {}',
                        self.file_data_csv)
        # update attribute
        self.heroku_data = df
        # return df with data
        return df

    def cb_to_coords(self, df, save_csv=True):
        """
        Create arrays with coordinates for images.
        save_points: save dictionary with points.
        if save_points:: save dictionary with points for each worker.
        """
        # load mapping of codes and coordinates
        with open(gz.common.get_configs('mapping_cb')) as f:
            mapping = json.load(f)
        # dictionaries to store points
        points = {}
        points_worker = {}
        # number of stimuli to process
        num_stimuli = gz.common.get_configs('num_stimuli')
        logger.info('Extracting coordinates for {} stimuli.', num_stimuli)
        # loop over stimuli from 1 to num_stimuli
        # tqdm adds progress bar
        for stim_id in tqdm(range(1, num_stimuli + 1)):
            # build names of columns in df
            image_cb = 'image_' + str(stim_id) + '-cb'
            image_in = 'image_' + str(stim_id) + '-in'
            # trim df
            stim_from_df = df[['group_choice',
                              image_cb,
                              image_in]]
            # replace nans with empty lists
            empty = pd.Series([[] for _ in range(len(stim_from_df.index))],
                              index=stim_from_df.index)
            stim_from_df[image_cb] = stim_from_df[image_cb].fillna(empty)
            stim_from_df[image_in] = stim_from_df[image_in].fillna(empty)
            # create empty list to store points for the stimulus
            points[stim_id] = []
            # iterate of data from participants for the given stimulus
            for pp in range(len(stim_from_df)):
                # input given by participant
                given_in = stim_from_df.iloc[pp][image_in]
                logger.debug('{}: from group {} found values {} input '
                             + 'for stimulus {}.',
                             stim_from_df.index[pp],
                             stim_from_df.iloc[pp]['group_choice'],
                             given_in,
                             stim_id)
                # iterate over all values given by the participand
                for val in range(len(given_in)):
                    # check if data from participant is present for the given
                    # stimulus
                    if (not stim_from_df.iloc[pp][image_cb][val] or
                       pd.isna(stim_from_df.iloc[pp][image_cb][val])):
                        # if no data present, move to the next participant
                        continue
                    # check if input is in mapping
                    mapping_cb = '../public/img/codeboard/cb_' + \
                                 stim_from_df.iloc[pp][image_cb][val] + \
                                 '.jpg'
                    if (given_in[val] in mapping[mapping_cb][1].keys()):
                        coords = mapping[mapping_cb][1][given_in[val]]
                        # add coordinates
                        if stim_id not in points:
                            points[stim_id] = [[int(coords[0]),
                                               int(coords[1])]]
                        else:
                            points[stim_id].append([int(coords[0]),
                                                    int(coords[1])])
                        if stim_from_df.index[pp] not in points_worker:
                            points_worker[stim_from_df.index[pp]] = [[int(coords[0]),  # noqa: E501
                                                                     int(coords[1])]]  # noqa: E501
                        else:
                            points_worker[stim_from_df.index[pp]].append([int(coords[0]),  # noqa: E501
                                                                          int(coords[1])])  # noqa: E501
                        # add coordinates
        # save to csv
        if save_csv:
            # create a dataframe to save to csv
            df_csv = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in points.items()]))  # noqa: E501
            df_csv = df_csv.transpose()
            # save to csv
            df_csv.to_csv(gz.settings.output_dir + '/' +
                          self.file_points_csv)
            logger.info('Saved points ditctionary to csv file {}',
                        self.file_points_csv)
            # create a dataframe to save to csv
            df_csv = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in points_worker.items()]))  # noqa: E501
            df_csv = df_csv.transpose()
            # save to csv
            df_csv.to_csv(gz.settings.output_dir + '/' +
                          self.file_points_worker_csv)
            logger.info('Saved points_worker ditctionary to csv file {}',
                        self.file_points_worker_csv)
        # return points
        return points, points_worker

    def filter_data(self, df):
        """
        Filter data based on the folllowing criteria:
            1. People who entered incorrect codes for sentinel images more than
               gz.common.get_configs('allowed_mistakes_sent') times.
        """
        # more than allowed number of mistake with codes for sentinel images
        # load mapping of codes and coordinates
        logger.info('Filtering heroku data.')
        logger.info('Filter-h1. People who made mistakes with sentinel '
                    + 'images.')
        with open(gz.common.get_configs('mapping_sentinel_cb')) as f:
            mapping = json.load(f)
        allowed_mistakes = gz.common.get_configs('allowed_mistakes_sent')
        # number of sentinel images in trainig
        training_total = gz.common.get_configs('training_sent')
        # df to store data to filter out
        df_1 = pd.DataFrame()
        # loop over rows in data
        # tqdm adds progress bar
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            # fill nans with empty lists
            empty = pd.Series([[] for _ in range(len(row.index))],
                              index=row.index)
            row = row.fillna(empty)
            # counter mistakes
            mistakes_counter = 0
            # counter sentinel images found in training
            training_counter = 0
            # loop over values in the row
            for index_r, value_r in row.iteritems():
                # check if input is given
                if (value_r == []):
                    # if no data present, move to the next cell
                    continue
                # sentinel image
                if 'sentinel_' in index_r and '-in' in index_r:
                    # sentinel image in training found
                    if training_counter < training_total:
                        # increase counter of sentinel images
                        training_counter = training_counter + 1
                        # skip since we are still in training data
                        continue
                    # sentinel image not in training found
                    else:
                        # increase counter of sentinel images
                        training_counter = training_counter + 1
                        sent_found = True
                        # extract ID of image
                        num_found = re.findall(r'\d+',
                                               index_r)
                        sent_name = num_found[0]
                        # check if input is in list of correct codes
                        mapping_cb = '../public/img/sentinel/sentinel_' + \
                                     str(sent_name) + \
                                     '.jpg'
                        if (value_r[0] not in mapping[mapping_cb]['correct_codes']):  # noqa: E501
                            # mistake found
                            mistakes_counter = mistakes_counter + 1
                            # check if limit was reached
                            if mistakes_counter > allowed_mistakes:
                                logger.debug('{}: found {} mistakes for '
                                             + 'sentinel images.',
                                             row['worker_code'],
                                             mistakes_counter)
                                # add to df with data to filter out
                                df_1 = df_1.append(row)
                                break
        logger.info('Filter-h1. People who made more than {} mistakes with '
                    + 'sentinel image: {}',
                    allowed_mistakes,
                    df_1.shape[0])
        # people that chose the coordiantes in the centre too often
        logger.info('Filter-h2. People who chose codeblocks in the middle too '
                    + 'often.')
        # df to store data to filter out
        df_2 = pd.DataFrame()
        # create arrays with coordinates for stimuli
        _, points_worker = self.cb_to_coords(df, False)
        # allowed percentage of codeblocks in the middle
        allowed_percentage = gz.common.get_configs('allowed_cb_middle')
        # get dimensions of stimulus
        width = gz.common.get_configs('stimulus_width')
        height = gz.common.get_configs('stimulus_height')
        area = gz.common.get_configs('cb_middle_area')
        # calculate the middle of the stimulus
        width_middle = round(width/2)
        height_middle = round(height/2)
        # ploygon for the centre
        polygon = Polygon([(width_middle - area, height_middle + area),
                           (width_middle + area, height_middle + area),
                           (width_middle - area, height_middle - area),
                           (width_middle + area, height_middle - area)])
        # detected percentage of codeblocks in the middle
        for worker_code, points in points_worker.items():
            # skip if no points for worker
            if len(points) == 0:
                continue
            detected = 0
            for point in points:
                # convert to point object
                point = Point(point[0], point[1])
                # check if point is within a polygon in the middle
                if polygon.contains(point):
                    # point in the middle detected
                    detected += 1
                    logger.debug('{}: point {} was in the middle polygon {}',
                                 worker_code,
                                 point,
                                 polygon)
            # Check if for th worker there were more than allowed limit of
            # points in the middle
            if detected / len(points) > allowed_percentage:
                logger.debug('{}: had {} share of points in the '
                             + 'middle polygon {}.',
                             worker_code,
                             detected,
                             polygon)
                df_2 = df_2.append(df[df['worker_code'] == worker_code])
        logger.info('Filter-h2. People who had more that {} share of '
                    + 'codeblocks in the middle: {}',
                    allowed_percentage,
                    df_2.shape[0])
        # concatanate dfs with filtered data
        old_size = df.shape[0]
        df_filtered = pd.concat([df_1, df_2])
        # drop rows with filtered data
        unique_worker_codes = df_filtered['worker_code'].drop_duplicates()
        df = df[~df['worker_code'].isin(unique_worker_codes)]
        logger.info('Filtered in total in heroku data: {}',
                    old_size - df.shape[0])
        return df
