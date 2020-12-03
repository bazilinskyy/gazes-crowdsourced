# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers

import gazes as gz

gz.logs(show_level='info', show_color=True)
logger = gz.CustomLogger(__name__)  # use custom logger


if __name__ == '__main__':
    # todo: add descriptions to methods
    # create object for working with heroku data
    files_heroku = gz.common.get_configs('files_heroku')
    heroku = gz.analysis.Heroku(files_data=files_heroku,
                                save_p=True,
                                load_p=False,
                                save_csv=True)
    # read heroku data
    heroku_data = heroku.read_data()
    # create object for working with appen data
    file_appen = gz.common.get_configs('file_appen')
    appen = gz.analysis.Appen(file_data=file_appen,
                              save_p=True,
                              load_p=False,
                              save_csv=True)
    # read appen data
    appen_data = appen.read_data()
    # get keys in data files
    heroku_data_keys = heroku_data.keys()
    appen_data_keys = appen_data.keys()
    # flag and reject cheaters
    qa = gz.qa.QA(file_cheaters=gz.common.get_configs('file_cheaters'),
                  job_id=gz.common.get_configs('appen_job'))
    # qa.flag_users()
    # qa.reject_users()
    # merge heroku and appen dataframes into one
    all_data = heroku_data.merge(appen_data,
                                 left_on='worker_code',
                                 right_on='worker_code')
    # update original data files
    heroku_data = all_data[all_data.columns.intersection(heroku_data_keys)]
    heroku_data = heroku_data.set_index('worker_code')
    heroku.set_data(heroku_data)  # update object with filtered data
    appen_data = all_data[all_data.columns.intersection(appen_data_keys)]
    appen_data = appen_data.set_index('worker_code')
    appen.set_data(appen_data)  # update object with filtered data
    # create arrays with coordinates for stimuli
    points = heroku.cb_to_coords()
    # create heatmaps
    analysis = gz.analysis.Analysis()
    # number of stimuli to process
    num_stimuli = gz.common.get_configs('num_stimuli')
    logger.info('Creating images with gazes and heatmaps for {} stimuli.',
                num_stimuli)
    # loop over stimuli from 1 to num_stimuli
    # tqdm adds progress bar
    for stim_id in tqdm(range(1, num_stimuli + 1)):
        # create path for stimulus
        stim_path = gz.common.get_configs('path_stimuli') + \
                    '/image_' + \
                    str(stim_id) + \
                    '.jpg'
        # create image with overlay of gazes for stimulus
        analysis.create_gazes(stim_path,
                              points[stim_id],
                              save_file=True)
        # create heatmaps for stimulus
        analysis.create_heatmap(stim_path,
                                points[stim_id],
                                type_heatmap='contourf',
                                add_corners=True,
                                save_file=True)
        analysis.create_heatmap(stim_path,
                                points[stim_id],
                                type_heatmap='pcolormesh',
                                add_corners=True,
                                save_file=True)
    # check if any figures are to be rendered
    figures = [manager.canvas.figure
               for manager in
               matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    # show figures, if any
    if figures:
        plt.show()
