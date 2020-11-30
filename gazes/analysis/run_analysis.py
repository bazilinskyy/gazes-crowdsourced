# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import json
import os
import matplotlib.pyplot as pltget_secrets

import gazes as gz

gz.logs(show_level='debug', show_color=True)
logger = gz.CustomLogger(__name__)  # use custom logger


def read_data(files_heroku, file_appen):
    # create object for working with heroku data
    heroku = gz.analysis.Heroku(files_data=files_heroku,
                                save_p=True,
                                load_p=False,
                                save_csv=True)
    # read heroku data
    heroku_data = heroku.read_data()
    # create object for working with appen data
    appen = gz.analysis.Appen(file_data=file_appen,
                              save_p=True,
                              load_p=False,
                              save_csv=True)
    # read heroku data
    appen_data = appen.read_data()
    # todo: filter data


if __name__ == '__main__':
    read_data(gz.common.get_configs('files_heroku'),
              gz.common.get_configs('file_appen'))  # read data
    analysis = gz.analysis.Analysis()
