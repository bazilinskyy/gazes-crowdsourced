# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import json
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import seaborn as sns
from scipy.stats.kde import gaussian_kde

import gazes as gz

matplotlib.use('TkAgg')
logger = gz.CustomLogger(__name__)  # use custom logger


class Analysis:
    def __init__(self):
        pass

    def create_gazes(self, image, points, save_file=False):
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
        # convert points into np array
        xy = np.array(points)
        x = xy[:, 0]
        y = xy[:, 1]
        # show heatmap by plt
        dpi = 150
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
        plt.imshow(im)
        for point in points:
            plt.plot(point[0],
                     point[1],
                     color='red',
                     marker='x',
                     markersize=10)
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
            self.save_fig(image, fig, '/figures/', '_gazes.jpg')

    def create_heatmap(self,
                       image,
                       points,
                       type_heatmap='contourf',  # contourf or pcolormesh
                       add_corners=True,
                       save_file=False):
        """
        Create heatmap for image based on the list of lists of points.
        add_corners: add points to the corners to have the heatmap ovelay the
                     whole image
        """
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
        except (np.linalg.LinAlgError, np.linalg.LinAlgError, ValueError) as e:
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
                plt.contourf(xi, yi, zi.reshape(xi.shape),
                             alpha=0.5)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
            except TypeError as e:
                logger.error('Not enough data. Heatmap was not created for '
                             + '{}.',
                             image)
                plt.close(fig)  # clear figure from memory
                return
            suffix_file = '_contourf.jpg'
        elif type_heatmap == 'pcolormesh':
            try:
                plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)
                plt.margins(0, 0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
            except TypeError as e:
                logger.error('Not enough data. Heatmap was not created for '
                             + '{}.',
                             image)
                plt.close(fig)  # clear figure from memory
                return
            suffix_file = '_pcolormesh.jpg'
        elif type_heatmap == 'kdeplot':
            try:
                sns.kdeplot(x, y, alpha=0.5, shade=True, cmap="RdBu_r")
            except TypeError as e:
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

    def create_histogram(self,
                         image,
                         points,
                         density_coef=10,
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
        # get dimensions of base image
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
            self.save_fig(image, fig, '/figures/', '_histogram.jpg')

    def save_fig(self, image, fig, output_subdir, suffix):
        # extract name of stimulus after last slash
        file_no_path = image.rsplit('/', 1)[-1]
        # remove extension
        file_no_path = os.path.splitext(file_no_path)[0]
        # create path
        path = gz.settings.output_dir + output_subdir
        if not os.path.exists(path):
            os.makedirs(path)
        # save file
        fig.tight_layout()
        plt.savefig(path + file_no_path + suffix,
                    bbox_inches='tight',
                    pad_inches=0)
        # clear figure from memory
        plt.close(fig)
