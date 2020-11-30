# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>

import json
import os
import matplotlib.pyplot as plt


class Heroku:
    files_data = []  # list of files with heroku data

    def __init__(self, files_data: list):
        self.files_data = files_data

    def read_data(self, files_heroku, file_appen):
        pass
