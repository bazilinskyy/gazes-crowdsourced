# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from tqdm import tqdm
import matplotlib.pyplot as plt

import gazes as gz

gz.logs(show_level='debug', show_color=True)
logger = gz.CustomLogger(__name__)  # use custom logger


if __name__ == '__main__':
    # create object for working with heroku data
    files_heroku = gz.common.get_configs('files_heroku')
    heroku = gz.analysis.Heroku(files_data=files_heroku,
                                save_p=False,
                                load_p=True,
                                save_csv=False)
    # read heroku data
    heroku_data = heroku.read_data()
    # create object for working with appen data
    file_appen = gz.common.get_configs('file_appen')
    appen = gz.analysis.Appen(file_data=file_appen,
                              save_p=False,
                              load_p=True,
                              save_csv=False)
    # read heroku data
    appen_data = appen.read_data()
    # todo: filter data

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
    # for stim_id in [1]:
        # create path for stimulus
        stim_path = gz.common.get_configs('path_stimuli') + \
                    '/image_' + \
                    str(stim_id) + \
                    '.jpg'
        # create image with overlay of gazes for stimulus
        analysis.create_gazes(stim_path,
                              points[stim_id],
                              save_file=True)
        # create heatmap for stimulus
        analysis.create_heatmap(stim_path,
                                points[stim_id],
                                save_file=True)
    # show images
    # plt.show()
