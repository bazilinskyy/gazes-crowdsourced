# by Pavlo Bazilinskyy <pavlo.bazilinskyy@gmail.com>
import json
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.stats.kde import gaussian_kde

import gazes as gz

matplotlib.use('TkAgg')


class Analysis:
    def __init__(self):
        pass

    def create_gazes(self, image, points, save_file=False):
        """
        Output gazes for image based on the list of lists of points.
        """
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
                   # extent=[x.min(), x.max(), y.min(), y.max()],
                   #aspect='auto')  # original image
        for point in points:
            plt.plot(point[0],
                     point[1],
                     color='red',
                     marker='x',
                     markersize=10)
        plt.axis('off')
        # save image
        if save_file:
            # extract name of stimulus after last slash
            file_no_path = image.rsplit('/', 1)[-1]
            # remove extension
            file_no_path = os.path.splitext(file_no_path)[0]
            # create path
            path = gz.settings.output_dir + '/heatmaps/'
            if not os.path.exists(path):
                os.makedirs(path)
            # save file
            fig.tight_layout()
            plt.savefig(path + file_no_path + '_gazes.png')

    def create_heatmap(self, image, points, save_file=False):
        """
        Create heatmap for image based on the list of lists of points.
        """
        # todo: check https://stackoverflow.com/questions/36957149/density-map-heatmaps-in-matplotlib
        # todo: implement smoothing https://stackoverflow.com/questions/2369492/generate-a-heatmap-in-matplotlib-using-a-scatter-data-set
        # get dimensions
        width = gz.common.get_configs('stimulus_width')
        height = gz.common.get_configs('stimulus_height')
        # convert points into np array
        xy = np.array(points)

        x = xy[:, 0]
        y = xy[:, 1]
        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,
                          y.min():y.max():y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # alpha=0.5 will make the plots semitransparent
        dpi = 150
        fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)

        # plt.pcolormesh(xi, yi, zi.reshape(xi.shape), alpha=0.5)
        plt.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.5)

        # plt.xlim(x.min(), x.max())
        # plt.ylim(y.min(), y.max())

        # read original image
        im = plt.imread(image)
        plt.imshow(im)
                   # extent=[x.min(), x.max(), y.min(), y.max()],
                   # aspect='auto')
        plt.axis('off')
        # save image
        if save_file:
            # extract name of stimulus after last slash
            file_no_path = image.rsplit('/', 1)[-1]
            # remove extension
            file_no_path = os.path.splitext(file_no_path)[0]
            # create path
            path = gz.settings.output_dir + '/heatmaps/'
            if not os.path.exists(path):
                os.makedirs(path)
            # save file
            fig.tight_layout()
            plt.savefig(path + file_no_path + '_heatmap.png')

    def visualise(self):
        pass
