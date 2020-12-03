# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import requests
import pandas as pd
from datetime import datetime

import gazes as gz

logger = gz.CustomLogger(__name__)  # use custom logger


class QA:
    file_cheaters = ''  # csv file with cheaters
    job_id = -1  # appen job ID

    def __init__(self,
                 file_cheaters: str,
                 job_id: int):
        self.file_cheaters = file_cheaters
        self.job_id = job_id

    def flag_users(self):
        """
        Flag users descibed in csv file self.file_cheaters from job
        self.job_id.
        """
        # import csv file
        df = pd.read_csv(self.file_cheaters)
        logger.info('Flagging {} users.', df.shape[0])
        # loop over users in the job
        for index, row in df.iterrows():
            # make a PUT request for flagging
            cmd_put = 'https://api.appen.com/v1/jobs/' + \
                      str(self.job_id) + \
                      '/workers/' + \
                      str(row['worker_id']) + \
                      '.json'
            if row['worker_code']:
                flag_text = 'User repeatidly ignored our instructions and ' \
                            + 'joined job from different accounts/IP ' \
                            + 'addresses. The same code ' \
                            + row['worker_code'] \
                            + ' used internally in the job was reused.'
            else:
                flag_text = 'User repeatidly ignored our instructions and ' \
                            + 'joined job from different accounts/IP ' \
                            + 'addresses. No worker code used internally  ' \
                            + 'was inputted (html regex validator was ' \
                            + 'bypassed).'
            params = {'flag': flag_text,
                      'key': gz.common.get_secrets('appen_api_key')}
            # send PUT request
            r = requests.put(cmd_put,
                             data=params)
            logger.info('Flagged user with message \'\'',
                        str(row['worker_id']),
                        + flag_text)
            # print status and returned message
            print(r, r.content)

    def reject_users(self):
        """
        Reject users descibed in csv file self.file_cheaters from job
        self.job_id.
        """
        # import csv file
        df = pd.read_csv(self.file_cheaters)
        logger.info('Rejecting {} users.', df.shape[0])
        # loop over users in the job
        for index, row in df.iterrows():
            # make a PUT request for flagging
            cmd_put = 'https://api.appen.com/v1/jobs/' + \
                      str(self.job_id) + \
                      '/workers/' + \
                      str(row['worker_id']) + \
                      '/reject.json'
            reason_text = 'User repeatidly ignored our instructions and ' \
                          + 'joined job from different accounts/IP ' \
                          + 'addresses. The same code ' \
                          + row['worker_code'] \
                          + ' used internally in the job was reused.'
            params = {'reason': reason_text,
                      'manual': 'true',
                      'key': gz.common.get_secrets('appen_api_key')}
            # send PUT request
            r = requests.put(cmd_put,
                             data=params)
            logger.info('Rejected user with message \'\'',
                        str(row['worker_id']),
                        + reason_text)
            # print status and returned message
            print(r, r.content)
