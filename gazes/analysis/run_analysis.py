# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import json
import os
import matplotlib.pyplot as pltget_secrets

import gazes as gz

# consts
# list of files with experiment data from heroku (can be multiple)
FILES_HEROKU = gz.common.get_configs('files_heroku')
# file from appen (can be only 1)
FILE_APPEN = gz.common.get_configs('file_appen')


def read_data(files_heroku, file_appen):
    # create object for working with heroku data
    heroku = gz.analysis.Heroku(files_heroku)
    # create object for working with appen data
    appen = gz.analysis.Appen(file_appen)


if __name__ == '__main__':
    read_data(FILES_HEROKU, FILE_APPEN)  # read data
    analysis = gz.analysis.Analysis()
