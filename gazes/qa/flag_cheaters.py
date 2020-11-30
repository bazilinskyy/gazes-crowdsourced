# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import gazes as gz
import requests
import pandas as pd
from datetime import datetime

# consts
API_KEY = gz.common.get_secrets('appen_api_key')

# print without timestamp
old_print = print


# print with timestamp
def timestamped_print(*args, **kwargs):
    old_print(datetime.now(), *args, **kwargs)


# assign print to have timestamps
print = timestamped_print


def flag_users(file_users, job_id):
    """Flag users descibed in csv file file_users from job job_id

    Args:
        file_users (string): users to flag
        job_id (string): appen job id
    """
    # import csv file
    df = pd.read_csv(file_users)
    # loop over users in the job
    for index, row in df.iterrows():
        # make a PUT request for flagging
        cmd_put = 'https://api.appen.com/v1/jobs/' + \
                  str(job_id) + \
                  '/workers/' + \
                  str(row['worker_id']) + \
                  '.json'
        flag_text = 'User repeatidly ignored our instructions and joined ' \
                    + 'job from different accounts/IP addresses. The same ' \
                    + 'code ' \
                    + row['worker_code'] \
                    + ' used internally in the job was reused.'
        params = {'flag': flag_text,
                  'key': API_KEY}
        # send PUT request
        print('Flag user '
              + str(row['worker_id'])
              + ' with message \"'
              + flag_text
              + '\"')
        r = requests.put(cmd_put,
                         data=params)
        # print status and returned message
        print(r, r.content)


def reject_users(file_users, job_id):
    """Reject users descibed in csv file file_users from job job_id

    Args:
        file_users (string): users to flag
        job_id (string): appen job id
    """
    # import csv file
    df = pd.read_csv(file_users)
    # loop over users in the job
    for index, row in df.iterrows():
        # make a PUT request for flagging
        cmd_put = 'https://api.appen.com/v1/jobs/' + \
                  str(job_id) + \
                  '/workers/' + \
                  str(row['worker_id']) + \
                  '/reject.json'
        reason_text = 'User repeatidly ignored our instructions and joined ' \
                      + 'job from different accounts/IP addresses. The same ' \
                      + 'code ' \
                      + row['worker_code'] \
                      + ' used internally in the job was reused.'
        params = {'reason': reason_text,
                  'manual': 'true',
                  'key': API_KEY}
        # send PUT request
        print('Reject user '
              + str(row['worker_id'])
              + ' with message \"'
              + reason_text
              + '\"')
        r = requests.put(cmd_put,
                         data=params)
        # print status and returned message
        print(r, r.content)


# flag lists of users
if __name__ == '__main__':
    flag_users('cheaters.csv', 1670745)  # flag users
    reject_users('cheaters.csv', 1670745)  # reject users
