# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers

import gazes as gz

gz.logs(show_level='info', show_color=True)
logger = gz.CustomLogger(__name__)  # use custom logger

# Const
SAVE_P = False  # save pickle files with data
LOAD_P = True  # load pickle files with data
SAVE_CSV = True  # load csv files with data
REJECT_CHEATERS = False  # reject cheaters on Appen
CALC_COORDS = False  # calculate coordinates (False saves time)
UPDATE_MAPPING = True  # update mapping with counts og gazes (False saves time)
file_coords = 'coords.p'  # file to save lists with coordinates
file_mapping = 'mapping.p'  # file to save lists with coordinates

if __name__ == '__main__':
    # todo: add descriptions for methods automatically with a sublime plugin
    # create object for working with heroku data
    files_heroku = gz.common.get_configs('files_heroku')
    heroku = gz.analysis.Heroku(files_data=files_heroku,
                                save_p=SAVE_P,
                                load_p=LOAD_P,
                                save_csv=SAVE_CSV)
    # read heroku data
    heroku_data = heroku.read_data()
    # create object for working with appen data
    file_appen = gz.common.get_configs('file_appen')
    appen = gz.analysis.Appen(file_data=file_appen,
                              save_p=SAVE_P,
                              load_p=LOAD_P,
                              save_csv=SAVE_CSV)
    # read appen data
    appen_data = appen.read_data()
    # get keys in data files
    heroku_data_keys = heroku_data.keys()
    appen_data_keys = appen_data.keys()
    # flag and reject cheaters
    if REJECT_CHEATERS:
        qa = gz.analysis.QA(file_cheaters=gz.common.get_configs('file_cheaters'),  # noqa: E501
                            job_id=gz.common.get_configs('appen_job'))
        qa.flag_users()
        qa.reject_users()
    # merge heroku and appen dataframes into one
    all_data = heroku_data.merge(appen_data,
                                 left_on='worker_code',
                                 right_on='worker_code')
    logger.info('Data from {} participants included in analysis.',
                all_data.shape[0])
    # update original data files
    heroku_data = all_data[all_data.columns.intersection(heroku_data_keys)]
    heroku_data = heroku_data.set_index('worker_code')
    heroku.set_data(heroku_data)  # update object with filtered data
    appen_data = all_data[all_data.columns.intersection(appen_data_keys)]
    appen_data = appen_data.set_index('worker_code')
    appen.set_data(appen_data)  # update object with filtered data
    appen.show_info()  # show info for filtered data
    # create arrays with coordinates for stimuli
    if CALC_COORDS:
        points, _, points_duration = heroku.cb_to_coords(heroku_data)
        gz.common.save_to_p(file_coords,
                            [points, points_duration],
                            'points data')
    else:
        points, points_duration = gz.common.load_from_p(file_coords,
                                                        'points data')
    # update mapping of stimuli
    if UPDATE_MAPPING:
        # read in mapping of stimuli
        stimuli_mapped = heroku.read_mapping()
        # populate coordinates in mapping of stimuli
        stimuli_mapped = heroku.populate_coords_mapping(heroku_data,
                                                        points_duration,
                                                        stimuli_mapped)
        gz.common.save_to_p(file_mapping,
                            stimuli_mapped,
                            'points data')
    else:
        stimuli_mapped = gz.common.load_from_p(file_mapping,
                                               'mapping of stimuli')
    # Output
    analysis = gz.analysis.Analysis()
    # plot gaze detections of vehicles for all stimuli
    analysis.detection_vehicle(stimuli_mapped, save_file=True)
    # create correlation matrix
    analysis.corr_matrix(stimuli_mapped, save_file=True)
    # number of stimuli to process
    num_stimuli = gz.common.get_configs('num_stimuli')
    logger.info('Creating figures for {} stimuli.', num_stimuli)
    # fetch stimulus durations
    durations = gz.common.get_configs('stimulus_durations')
    # loop over stimuli from 1 to num_stimuli
    # tqdm adds progress bar
    for stim_id in tqdm(range(1, num_stimuli + 1)):
        # create path for stimulus
        stim_path = gz.common.get_configs('path_stimuli') + \
                    '/image_' + \
                    str(stim_id) + \
                    '.jpg'
        # check if data for the stimulus is present
        if stim_id not in points.keys():
            continue
        # create image with overlay of gazes for stimulus
        analysis.create_gazes(stim_path,
                              points[stim_id],
                              save_file=True)
        # create histogram for stimulus and durations
        points_process = {}
        for points_dur in range(len(points_duration)):
            suffix = '_gazes_' + str(durations[points_dur]) + '.jpg'
            analysis.create_gazes(stim_path,
                                  points_duration[points_dur][stim_id],
                                  suffix=suffix,
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
        analysis.create_heatmap(stim_path,
                                points[stim_id],
                                type_heatmap='kdeplot',
                                add_corners=True,
                                save_file=True)
        # create histogram for stimulus
        analysis.create_histogram(stim_path,
                                  points[stim_id],
                                  density_coef=20,
                                  save_file=True)
        # create histogram for stimulus and durations
        points_process = {}
        for points_dur in range(len(points_duration)):
            suffix = '_histogram_' + str(durations[points_dur]) + '.jpg'
            analysis.create_histogram(stim_path,
                                      points_duration[points_dur][stim_id],
                                      density_coef=20,
                                      suffix=suffix,
                                      save_file=True)
        # create animation for stimulus
        points_process = {}
        for points_dur in range(len(points_duration)):
            points_process[points_dur] = points_duration[points_dur][stim_id]
        analysis.create_animation(stim_path,
                                  stim_id,
                                  points_process,
                                  save_anim=True,
                                  save_frames=True)
        # plot gaze detections of vehicles
        analysis.detection_vehicle_image(stimuli_mapped,
                                         stim_path,
                                         stim_id,
                                         save_file=True)
        # draw polygon on top on base image
        analysis.draw_polygon(stim_path,
                              stim_id,
                              save_file=True)
    # stitch animations into 1 long videos
    analysis.create_animation_all_stimuli(num_stimuli)
    # check if any figures are to be rendered
    figures = [manager.canvas.figure
               for manager in
               matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
    # show figures, if any
    if figures:
        plt.show()
