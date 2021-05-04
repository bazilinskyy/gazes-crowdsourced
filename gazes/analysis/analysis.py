# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import os
import subprocess
import io
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats.kde import gaussian_kde

import gazes as gz

matplotlib.use('TkAgg')
logger = gz.CustomLogger(__name__)  # use custom logger


class Analysis:
    # used by animation
    fig = None
    g = None
    image = None
    stim_id = None
    points = None
    save_frames = False
    folder = '/figures/'
    polygons = None

    def __init__(self):
        # read mapping of polygons from a csv file
        self.polygons = pd.read_csv(gz.common.get_configs('vehicles_polygons'))
        # set index as stimulus_id
        self.polygons.set_index('image_id', inplace=True)
        # set font to Times
        plt.rc('font', family='serif')

    def create_gazes(self,
                     image,
                     points,
                     suffix='_gazes.jpg',
                     save_file=False):
        """
        Output gazes for image based on the list of lists of points.
        """
        # check if data is present
        if not points:
            logger.error('Not enough data. Gazes visualisation was not '
                         + 'created for {}.', image)
            return
        # read original image
        im = plt.imread(image)
        # get dimensions
        width = gz.common.get_configs('stimulus_width')
        height = gz.common.get_configs('stimulus_height')
        # show heatmap by plt
        dpi = 150
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        plt.imshow(im)
        for point in points:
            plt.plot(point[0],
                     point[1],
                     color='red',
                     marker='x',
                     markersize=1)
        # remove white spaces around figure
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1,
                            bottom=0,
                            right=1,
                            left=0,
                            hspace=0,
                            wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # save image
        if save_file:
            self.save_fig(image, fig, self.folder, suffix)

    def create_heatmap(self,
                       image,
                       points,
                       type_heatmap='contourf',
                       add_corners=True,
                       save_file=False):
        """
        Create heatmap for image based on the list of lists of points.
        add_corners: add points to the corners to have the heatmap ovelay the
                     whole image
        type_heatmap: contourf, pcolormesh, kdeplot
        """
        # todo: remove datapoints in corners in heatmaps
        # check if data is present
        if not points:
            logger.error('Not enough data. Heatmap was not created for {}.',
                         image)
            return
        # get dimensions of base image
        width = gz.common.get_configs('stimulus_width')
        height = gz.common.get_configs('stimulus_height')
        # add datapoints to corners for maximised heatmaps
        if add_corners:
            if [0, 0] not in points:
                points.append([0, 0])
            if [width, height] not in points:
                points.append([width - 1, height - 1])
        # convert points into np array
        xy = np.array(points)
        # split coordinates list for readability
        x = xy[:, 0]
        y = xy[:, 1]
        # compute data for the heatmap
        try:
            k = gaussian_kde(np.vstack([x, y]))
            xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,
                              y.min():y.max():y.size**0.5*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        except (np.linalg.LinAlgError, np.linalg.LinAlgError, ValueError):
            logger.error('Not enough data. Heatmap was not created for {}.',
                         image)
            return
        # create figure object with given dpi and dimensions
        dpi = 150
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        # alpha=0.5 makes the plot semitransparent
        suffix_file = ''  # suffix to add to saved image
        if type_heatmap == 'contourf':
            try:
                g = plt.contourf(xi, yi, zi.reshape(xi.shape),
                                 alpha=0.5)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
            except TypeError:
                logger.error('Not enough data. Heatmap was not created for '
                             + '{}.',
                             image)
                plt.close(fig)  # clear figure from memory
                return
            suffix_file = '_contourf.jpg'
        elif type_heatmap == 'pcolormesh':
            try:
                g = plt.pcolormesh(xi, yi, zi.reshape(xi.shape),
                                   shading='auto',
                                   alpha=0.5)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
            except TypeError:
                logger.error('Not enough data. Heatmap was not created for '
                             + '{}.',
                             image)
                plt.close(fig)  # clear figure from memory
                return
            suffix_file = '_pcolormesh.jpg'
        elif type_heatmap == 'kdeplot':
            try:
                g = sns.kdeplot(x=x,
                                y=y,
                                alpha=0.5,
                                shade=True,
                                cmap="RdBu_r")
            except TypeError:
                logger.error('Not enough data. Heatmap was not created for '
                             + '{}.',
                             image)
                fig.clf()  # clear figure from memory
                return
            suffix_file = '_kdeplot.jpg'
        else:
            logger.error('Wrong type_heatmap {} given.', type_heatmap)
            plt.close(fig)  # clear from memory
            return
        # read original image
        im = plt.imread(image)
        plt.imshow(im)
        # remove axis
        plt.gca().set_axis_off()
        # remove white spaces around figure
        plt.subplots_adjust(top=1,
                            bottom=0,
                            right=1,
                            left=0,
                            hspace=0,
                            wspace=0)
        # save image
        if save_file:
            self.save_fig(image, fig, '/figures/', suffix_file)
        # return graph objects
        return fig, g

    def create_histogram(self,
                         image,
                         points,
                         density_coef=10,
                         suffix='_histogram.jpg',
                         save_file=False):
        """
        Create histogram for image based on the list of lists of points.
        density_coef: coeficient for division of dimensions for density of
                        points.
        """
        # check if data is present
        if not points:
            logger.error('Not enough data. Histogram was not created for {}.',
                         image)
            return
        # get dimensions of stimulus
        width = gz.common.get_configs('stimulus_width')
        height = gz.common.get_configs('stimulus_height')
        # convert points into np array
        xy = np.array(points)
        # split coordinates list for readability
        x = xy[:, 0]
        y = xy[:, 1]
        # create figure object with given dpi and dimensions
        dpi = 150
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        # build histogram
        plt.hist2d(x=x,
                   y=-y,  # convert to the reference system in image
                   bins=[round(width/density_coef),
                         round(height/density_coef)],
                   cmap=plt.cm.jet)
        plt.colorbar()
        # remove white spaces around figure
        plt.gca().set_axis_off()
        # save image
        if save_file:
            self.save_fig(image, fig, self.folder, suffix)

    def create_animation(self,
                         image,
                         stim_id,
                         points,
                         save_anim=False,
                         save_frames=False):
        """
        Create animation for image based on the list of lists of points of
        varying duration.
        """
        self.image = image
        self.stim_id = stim_id
        self.points = points
        self.save_frames = save_frames
        self.fig, self.g = self.create_heatmap(image,
                                               points[0],
                                               type_heatmap='kdeplot',  # noqa: E501
                                               add_corners=True,  # noqa: E501
                                               save_file=False)
        anim = animation.FuncAnimation(self.fig,
                                       self.animate,
                                       frames=len(points),
                                       interval=1000,
                                       repeat=False)
        # save image
        if save_anim:
            self.save_anim(image, anim, self.folder, '_animation.mp4')

    def create_animation_all_stimuli(self, num_stimuli):
        """
        Create long video with all animations.
        """
        logger.info('Creating long video with all animations for {} stimuli.',
                    num_stimuli)
        # create path
        path = gz.settings.output_dir + self.folder
        if not os.path.exists(path):
            os.makedirs(path)
        # file with list of animations
        list_anim = path + 'animations.txt'
        file = open(list_anim, 'w+')
        # loop of stimuli
        for stim_id in range(1, num_stimuli + 1):
            # add animation to the list
            anim_path = path + 'image_' + str(stim_id) + '_animation.mp4'
            # check if need to add a linebreak
            if stim_id == num_stimuli:
                file.write('file ' + anim_path)  # no need for linebreak
            else:
                file.write('file ' + anim_path + '\n')
        # close file with animations
        file.close()
        # stitch videos together
        os.chdir(path)
        subprocess.call(['ffmpeg',
                         '-y',
                         '-loglevel', 'quiet',
                         '-f', 'concat',
                         '-safe', '0',
                         '-i', list_anim,
                         '-c', 'copy',
                         'all_animations.mp4'])
        # delete file with animations
        os.remove(list_anim)

    def animate(self, i):
        """
        Helper function to create animation.
        """
        self.g.clear()
        self.g = sns.kdeplot(x=[item[0] for item in self.points[i]],
                             y=[item[1] for item in self.points[i]],
                             alpha=0.5,
                             shade=True,
                             cmap='RdBu_r')
        # read original image
        im = plt.imread(self.image)
        plt.imshow(im)
        # remove axis
        plt.gca().set_axis_off()
        # remove white spaces around figure
        plt.subplots_adjust(top=1,
                            bottom=0,
                            right=1,
                            left=0,
                            hspace=0,
                            wspace=0)
        # textbox with duration
        durations = gz.common.get_configs('stimulus_durations')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.75,
                 0.98,
                 'id=' + str(self.stim_id) + ' duration=' + str(durations[i]),
                 transform=plt.gca().transAxes,
                 fontsize=12,
                 verticalalignment='top',
                 bbox=props)
        # save each frame as file
        if self.save_frames:
            # build suffix for filename
            suffix = '_kdeplot_' + str(durations[i]) + '.jpg'
            # copy figure in buffer to prevent distruction of object
            buf = io.BytesIO()
            pickle.dump(self.fig, buf)
            buf.seek(0)
            temp_fig = pickle.load(buf)
            # save figure
            self.save_fig(self.image, temp_fig, self.folder, suffix)
        return self.g

    def detection_vehicle(self, mapping, save_file=False):
        """
        Detections of vehicles for stimuli for all images.
        """
        # stimulus durations
        durations = gz.common.get_configs('stimulus_durations')
        # set larger font
        s_font = 14  # small
        m_font = 18  # medium
        l_font = 20  # large
        plt.rc('font', size=s_font)         # controls default text sizes
        plt.rc('axes', titlesize=s_font)    # fontsize of the axes title
        plt.rc('axes', labelsize=m_font)    # fontsize of the axes labels
        plt.rc('xtick', labelsize=s_font)   # fontsize of the tick labels
        plt.rc('ytick', labelsize=s_font)   # fontsize of the tick labels
        plt.rc('legend', fontsize=s_font)   # fontsize of the legend
        plt.rc('figure', titlesize=l_font)  # fontsize of the figure title
        plt.rc('axes', titlesize=m_font)    # fontsize of the subplot title
        # create subplot
        fig, ax = plt.subplots(2,  # rows
                               2,  # columns
                               figsize=(15, 11))  # width, height in inches
        # settings for subplots
        # todo: automatic upper limit
        ylim = [0, 3600]
        bar_width = 0.8
        xticks_angle = 45
        # 1. distance
        # get data
        df_plot = mapping.groupby(['dist']).sum(numeric_only=True)
        # build
        df_plot[durations].transpose().plot.bar(stacked=True,
                                                ax=ax[0, 0],
                                                width=bar_width)
        # axis labels
        ax[0, 0].set_ylabel('Number of gazes on target vehicle')
        # ticks
        ax[0, 0].tick_params(axis='x', labelrotation=xticks_angle)
        # assign labels
        self.autolabel(ax[0, 0], on_top=False, decimal=False)
        # legend
        ax[0, 0].legend([r'$dist = 0$',
                         r'$dist = 1$',
                         r'$dist = 2$'],
                        loc='upper left',
                        ncol=3)
        # title
        ax[0, 0].title.set_text('(a) Distance to vehicle')
        # grid lines
        ax[0, 0].grid(True, axis='y')
        ax[0, 0].set_axisbelow(True)
        # limits
        ax[0, 0].set_ylim(ylim)
        # 2. traffic
        # get data
        df_plot = mapping.groupby(['traf']).sum(numeric_only=True)
        # build
        df_plot[durations].transpose().plot.bar(stacked=True,
                                                ax=ax[0, 1],
                                                width=bar_width)
        # ticks
        ax[0, 1].tick_params(axis='x', labelrotation=xticks_angle)
        # assign labels
        self.autolabel(ax[0, 1], on_top=False, decimal=False)
        # legend
        ax[0, 1].legend([r'$traf=0$', r'$traf=1$', r'$traf=2$'],
                        loc='upper left',
                        ncol=3)
        # title
        ax[0, 1].title.set_text('(b) Traffic density')
        # grid lines
        ax[0, 1].grid(True, axis='y')
        ax[0, 1].set_axisbelow(True)
        # limits
        ax[0, 1].set_ylim(ylim)
        # 3. clutter
        # get data
        df_plot = mapping.groupby(['clut']).sum(numeric_only=True)
        # build
        df_plot[durations].transpose().plot.bar(stacked=True,
                                                ax=ax[1, 0],
                                                width=bar_width)
        # axis labels
        ax[1, 0].set_xlabel('Stimulus duration (ms)')
        ax[1, 0].set_ylabel('Number of gazes on target vehicle')
        # ticks
        ax[1, 0].tick_params(axis='x', labelrotation=xticks_angle)
        # assign labels
        self.autolabel(ax[1, 0], on_top=False, decimal=False)
        # legend
        ax[1, 0].legend([r'$clut=0$', r'$clut=1$', r'$clut=2$'],
                        loc='upper left',
                        ncol=3)
        # title
        ax[1, 0].title.set_text('(c) Visual clutter')
        # grid lines
        ax[1, 0].grid(True, axis='y')
        ax[1, 0].set_axisbelow(True)
        # limits
        ax[1, 0].set_ylim(ylim)
        # 4. vehicle type
        # get data
        df_plot = mapping.groupby(['veh']).sum(numeric_only=True)
        # build
        df_plot[durations].transpose().plot.bar(stacked=True,
                                                ax=ax[1, 1],
                                                width=bar_width)
        # axis labels
        ax[1, 1].set_xlabel('Stimulus duration (ms)')
        # ticks
        ax[1, 1].tick_params(axis='x', labelrotation=xticks_angle)
        # assign labels
        self.autolabel(ax[1, 1], on_top=False, decimal=False)
        # legend
        ax[1, 1].legend([r'$veh = 0$',
                         r'$veh = 1$'],
                        loc='upper left',
                        ncol=2)
        # title
        ax[1, 1].title.set_text('(d) Vehicle type')
        # grid lines
        ax[1, 1].grid(True, axis='y')
        ax[1, 1].set_axisbelow(True)
        # limits
        ax[1, 1].set_ylim(ylim)
        # tight layout
        fig.tight_layout()
        # remove white spaces around figure
        fig.subplots_adjust(top=0.97,
                            bottom=0.086,
                            right=0.98,
                            left=0.048,
                            hspace=0.286,
                            wspace=0.15)
        # save image
        if save_file:
            self.save_fig('all',
                          fig,
                          self.folder,
                          '_gazes_vehicle.jpg',
                          pad_inches=0.05)
        # revert font
        self.reset_font()

    def detection_vehicle_image(self,
                                mapping,
                                image,
                                stim_id,
                                save_file=False):
        """
        Detections of vehicles for stimuli for individual image.
        """
        # stimulus durations
        durations = gz.common.get_configs('stimulus_durations')
        # limit to given image
        mapping = mapping[mapping.index == stim_id]
        # create figure
        fig = plt.figure()
        # get sums of gazes
        df_plot = mapping[durations].sum(numeric_only=True)
        ax = df_plot.plot(kind='bar', width=0.8)
        # axis labels
        ax.set_xlabel('Stimulus duration (ms)')
        ax.set_ylabel('Number of gazes on target vehicle')
        # ticks
        ax.tick_params(axis='x', labelrotation=45)
        # assign labels
        self.autolabel(ax, on_top=True, decimal=False)
        # grid lines
        ax.grid(True, axis='y')
        ax.set_axisbelow(True)
        # tight layout
        fig.tight_layout()
        # remove white spaces around figure
        fig.subplots_adjust(top=0.97,
                            bottom=0.086,
                            right=0.98,
                            left=0.048,
                            hspace=0.286,
                            wspace=0.1)
        # save image
        if save_file:
            self.save_fig(image,
                          fig,
                          self.folder,
                          '_gazes_vehicle.jpg',
                          pad_inches=0.05)

    def corr_matrix(self, mapping, save_file=False):
        """
        Output correlation matrix.
        """
        # drop time and group_2 columns
        durations = gz.common.get_configs('stimulus_durations')
        columns_drop = ['time', 'group_2', 'gazes_mean']
        for dur in durations:
            columns_drop.append(str(dur) + '_count')
        mapping = mapping.drop(columns_drop, 1)
        mapping.fillna(0, inplace=True)
        # create correlation matrix
        corr = mapping.corr()
        # create mask
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        # set larger font
        s_font = 12  # small
        m_font = 16  # medium
        l_font = 18  # large
        plt.rc('font', size=s_font)         # controls default text sizes
        plt.rc('axes', titlesize=s_font)    # fontsize of the axes title
        plt.rc('axes', labelsize=m_font)    # fontsize of the axes labels
        plt.rc('xtick', labelsize=s_font)   # fontsize of the tick labels
        plt.rc('ytick', labelsize=s_font)   # fontsize of the tick labels
        plt.rc('legend', fontsize=s_font)   # fontsize of the legend
        plt.rc('figure', titlesize=l_font)  # fontsize of the figure title
        plt.rc('axes', titlesize=m_font)    # fontsize of the subplot title
        # create figure
        fig = plt.figure(figsize=(15, 8))
        g = sns.heatmap(corr,
                        annot=True,
                        mask=mask,
                        cmap='coolwarm',
                        fmt=".2f")
        # rotate ticks
        for item in g.get_xticklabels():
            item.set_rotation(45)
        # save image
        if save_file:
            self.save_fig('all',
                          fig,
                          self.folder,
                          '_corr_matrix.jpg',
                          pad_inches=0.05)
        # revert font
        self.reset_font()

    def draw_polygon(self, image, stim_id, save_file=False):
        # polygon of vehicle
        coords = np.array(self.polygons.at[stim_id, 'coords'].split(','),
                          dtype=int).reshape(-1, 2)
        # repeat the first point to create a 'closed loop'
        coords = np.append(coords, coords[0])
        coords = coords.reshape(-1, 2)
        # create lists of x and y values
        xs, ys = zip(*coords)
        # get dimensions of stimulus
        width = gz.common.get_configs('stimulus_width')
        height = gz.common.get_configs('stimulus_height')
        # create figure object with given dpi and dimensions
        dpi = 150
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        # draw polygon
        plt.plot(xs, ys, color='red')
        # read original image
        im = plt.imread(image)
        plt.imshow(im)
        # remove axis
        plt.gca().set_axis_off()
        # remove white spaces around figure
        plt.subplots_adjust(top=1,
                            bottom=0,
                            right=1,
                            left=0,
                            hspace=0,
                            wspace=0)
        # save image
        if save_file:
            self.save_fig(image, fig, self.folder, '_polygon.jpg')

    def save_fig(self, image, fig, output_subdir, suffix, pad_inches=0):
        """
        Helper function to save figure as file.
        """
        # extract name of stimulus after last slash
        file_no_path = image.rsplit('/', 1)[-1]
        # remove extension
        file_no_path = os.path.splitext(file_no_path)[0]
        # create path
        path = gz.settings.output_dir + output_subdir
        if not os.path.exists(path):
            os.makedirs(path)
        # save file
        plt.savefig(path + file_no_path + suffix,
                    bbox_inches='tight',
                    pad_inches=pad_inches)
        # clear figure from memory
        plt.close(fig)

    def save_anim(self, image, anim, output_subdir, suffix):
        """
        Helper function to save figure as file.
        """
        # extract name of stimulus after last slash
        file_no_path = image.rsplit('/', 1)[-1]
        # remove extension
        file_no_path = os.path.splitext(file_no_path)[0]
        # create path
        path = gz.settings.output_dir + output_subdir
        if not os.path.exists(path):
            os.makedirs(path)
        # save file
        anim.save(path + file_no_path + suffix, writer='ffmpeg')
        # clear animation from memory
        plt.close(self.fig)

    def autolabel(self, ax, on_top=False, decimal=True):
        """
        Attach a text label above each bar in *rects*, displaying its height.
        """
        # todo: optimise to use the same method
        # on top of bar
        if on_top:
            for rect in ax.patches:
                height = rect.get_height()
                # show demical points
                if decimal:
                    label_text = f'{height:.2f}'
                else:
                    label_text = f'{height:.0f}'
                ax.annotate(label_text,
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center',
                            va='bottom')
        # in the middle of the bar
        else:
            # based on https://stackoverflow.com/a/60895640/46687
            # .patches is everything inside of the chart
            for rect in ax.patches:
                # Find where everything is located
                height = rect.get_height()
                width = rect.get_width()
                x = rect.get_x()
                y = rect.get_y()
                # The height of the bar is the data value and can be used as
                # the label
                # show demical points
                if decimal:
                    label_text = f'{height:.2f}'
                else:
                    label_text = f'{height:.0f}'
                label_x = x + width / 2
                label_y = y + height / 2
                # plot only when height is greater than specified value
                if height > 0:
                    ax.text(label_x,
                            label_y,
                            label_text,
                            ha='center',
                            va='center')

    def reset_font(self):
        """
        Reset font to default size values. Info at
        https://matplotlib.org/tutorials/introductory/customizing.html
        """
        s_font = 8
        m_font = 10
        l_font = 12
        plt.rc('font', size=s_font)         # controls default text sizes
        plt.rc('axes', titlesize=s_font)    # fontsize of the axes title
        plt.rc('axes', labelsize=m_font)    # fontsize of the axes labels
        plt.rc('xtick', labelsize=s_font)   # fontsize of the tick labels
        plt.rc('ytick', labelsize=s_font)   # fontsize of the tick labels
        plt.rc('legend', fontsize=s_font)   # legend fontsize
        plt.rc('figure', titlesize=l_font)  # fontsize of the figure title
