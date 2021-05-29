"""Base class for specific mixture models both included in
sklearn and developed using the sklearn algorithms."""

# Author: Colin Weber
# Contact: colin.weber.27@gmail.com
# License: MIT
# sklearn License: BSD 3 clause

from abc import ABCMeta, abstractmethod

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image, ImageTk
from tkinter import Tk

from ._cluster_merger_gui import ClusterMergerGUI


def sig_fig_rounder(data: list, err: list):
    """Rounds the data based on its standard error according to
    standard significant figure rules.

    Parameters
    ----------
    data : list, len(n_samples)
        The data to be rounded.

    err : list, len(n_samples)
        The standard error in the data.
    """
    for n in range(len(err)):
        if err[n] <= 0:
            err[n] = 0.001  # Ensure all errors are positive
        err_string = str(err[n])
        err_string = err_string.replace('0', '')
        err_string = err_string.replace('.', '')
        data_string = str(data[n])
        data_string = data_string.replace('0', '')
        data_string = data_string.replace('.', '')
        if data_string[-1] == '5':
            data_string = str(data[n]) + '1'
            data[n] = float(data_string)
        if err_string[0] != '1':
            if err_string[-1] == '5':
                err_string = str(err[n]) + '1'
                err[n] = float(err_string)
            decimal_places = -int(np.floor(np.log10(err[n])))
            data[n] = str(round(data[n], decimal_places))
            err[n] = str(round(err[n], decimal_places))
            if len(data[n]) < abs(decimal_places) + 2 and \
                    data[n][0] != '-':
                while len(data[n]) != abs(decimal_places) + 2:
                    data[n] += '0'
            if len(data[n]) < abs(decimal_places) + 3 and \
                    data[n][0] == '-':
                while len(data[n]) != abs(decimal_places) + 3:
                    data[n] += '0'
            if len(err[n]) < abs(decimal_places) + 2 and \
                    err[n][-1] != '2':
                while len(err[n]) != abs(decimal_places) + 2:
                    err[n] += '0'
            if float(err[n]) >= 2:
                data[n] = str(int(round(float(data[n]), 0)))
                err[n] = str(int(round(float(err[n]), 0)))
        else:
            if err_string[-1] == '5':
                err_string = str(err[n]) + '1'
                err[n] = float(err_string)
            decimal_places = 1 - int(np.floor(np.log10(err[n])))
            data[n] = str(round(data[n], decimal_places))
            err[n] = str(round(err[n], decimal_places))
            test_err = err[n].replace('0', '')
            test_err = test_err.replace('.', '')
            if test_err[0] == '2':
                decimal_places -= 1
            if len(data[n]) < abs(decimal_places) + 2 and \
                    data[n][0] != '-':
                while len(data[n]) != abs(decimal_places) + 2:
                    data[n] += '0'
            if len(data[n]) < abs(decimal_places) + 3 and \
                    data[n][0] == '-':
                while len(data[n]) != abs(decimal_places) + 3:
                    data[n] += '0'
            if len(err[n]) < abs(decimal_places) + 2 and \
                    err[n][-1] != '2':
                while len(err[n]) != abs(decimal_places) + 2:
                    err[n] += '0'
            if float(err[n]) >= 2:
                data[n] = str(int(round(float(data[n]), 0)))
                err[n] = str(int(round(float(err[n]), 0)))

    return data, err


class GaussianMixtureBase(metaclass=ABCMeta):
    """Base class for specific mixture models both included in
    sklearn and developed using the sklearn algorithms.

    It specifies an interface for the other classes and provides
    basic common methods.

    version : 0.1
    """
    def __init__(self, n_components: int, cov_type: str, tol: float, max_iter: int,
                 n_init: int):
        self.n_components = n_components
        self.cov_type = cov_type
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.means_ = None
        self.weights_ = None
        self.covariances_ = None
        self.labels_ = None
        self.responsibilities_ = None
        self.n_comps_found_ = None
        self.centers_array_ = None
        self.ips_ = None
        self.unique_labels_ = None
        self.colors_ = None
        self.noise_colors_ = None
        self.clustered_ = False

    def _check_base_parameters(self):
        """Check values of the base's parameters.
        """
        if self.n_components < 1:
            raise ValueError("Invalid value for 'n_components': %d "
                             "Estimation requires at least one component"
                             % self.n_components)

        if self.tol < 0.:
            raise ValueError("Invalid value for 'tol': %.5f "
                             "Tolerance used by the EM must be non-negative"
                             % self.tol)

        if self.n_init < 1:
            raise ValueError("Invalid value for 'n_init': %d "
                             "Estimation requires at least one run"
                             % self.n_init)

        if self.max_iter < 1:
            raise ValueError("Invalid value for 'max_iter': %d "
                             "Estimation requires at least one iteration"
                             % self.max_iter)

        # Check all the parameters values of the derived class
        self._check_parameters()

    @abstractmethod
    def _check_parameters(self):
        """Check initial parameters of the derived class."""
        pass

    @abstractmethod
    def _calc_secondary_centers_unc(self, c1s, c1s_err, c2s,
                                    c2s_err, data_frame_object: object):
        """Calculate the coordinates of the cluster centers for the coordinate system
        that was not used for the fit.

        Standard errors are calculated with typical standard error propagation methods.
        Assigns the attributes 'centers_array_' and 'noise_colors_'.

        Parameters
        ----------
        c1s : array-like, shape (n_components,)
            The x-coordinates of the cluster centers if clustering
            with Cartesian coordinates, otherwise the r-coordinates
            of the cluster centers.

        c1s_err : array-like, shape (n_components,)
            The standard error in the c1s.

        c2s : array-like, shape (n_components,)
            The y-coordinates of the cluster centers if clustering
            with Cartesian coordinates, otherwise the p-coordinates
            of the cluster centers.

        c2s_err : array-like, shape (n_components,)
            The standard error in the c2s.

        data_frame_object : DataFrame class object
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.
        """
        pass

    @abstractmethod
    def _calculate_centers_uncertainties(self, data_frame_object: object):
        """After clustering the data, organize the cluster centers into a more accessible format.

        Assigns the attributes 'centers_array_', 'ips_',
        'unique_labels', colors_', and 'noise_colors_'.

        Parameters
        ----------
        data_frame_object : DataFrame class object
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.
        """
        pass

    @abstractmethod
    def recalculate_centers_uncertainties(self, data_frame_object: object, indices=None):
        """Recalculate the centers of each cluster and the uncertainties in the centers.

        This uses a different method from simply extracting the centers
        and uncertainties from the fit. Instead, it fits a univariate Gaussian
        to each dimension of each cluster and uses the statistics from the
        fits to calculate the centers. It assigns the attributes 'centers_array_'
        and 'noise_colors_'. This method was written by Dwaipayan
        Ray and Adrian Valverde.

        Parameters
        ----------
        data_frame_object : DataFrame class object
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.

        indices : list (optional)
            A list of the indices corresponding to the cluster centers to recalculate.
        """
        pass

    @abstractmethod
    def _cluster_data_one_d(self, x):
        """Use the Gaussian Mixture fits from the sklearn package to cluster the data.

        Assigns the object the attributes 'means_',
        'covariances_', 'weights_', 'labels_', 'responsibilities_',
        and 'n_comps_found_'.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_attributes)
            The data to be clustered.
        """
        pass

    @abstractmethod
    def cluster_data(self, data_frame_object: object):
        """Use the Gaussian Mixture fits from the sklearn package to cluster the data.

        Assigns the object the attributes 'means_',
        'covariances_', 'weights_', 'labels_', 'responsibilities_',
        and 'n_comps_found_'.

        Parameters
        ----------
        data_frame_object : DataFrame class object
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.
        """
        pass

    @abstractmethod
    def cluster_data_strict(self, data_frame_object: object):
        """Cluster the data, but restrict n_components to the value of the parameter 'n_components'.

        Assigns the mixture object the attributes 'means_',
        'covariances_', 'weights_', 'labels_', 'responsibilities_',
        and 'n_comps_found'.

        Parameters
        ----------
        data_frame_object : DataFrame class object
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.
        """
        pass

    def _identify_noise_colors(self):
        """Identify the colors of noise clusters.

        Noise clusters, loosely speaking, are clusters which we can't conclude are
        composed of identical ion species. Visually, these clusters can have large
        uncertainties, fewer ions, or 'span' other clusters, meaning they have samples
        that are on either side of another cluster. Although it's important to still
        have these clusters in our data set, it can still be helpful to mark them if we
        can. One way of doing this is with the 500% rules. We found that we can
        identify 60% of noise clusters if we define a noise cluster as one whose
        standard error in the center spot is 500% higher than the weighted average of
        the standard errors of the center spots in a data set, with the weights
        given by the number of samples in the cluster. This method applies that rule
        to identify some, but not all, of the noise clusters, which it identifies by
        color. It assigns the attribute 'noise_colors_'.
        """
        colors = ['blue', 'salmon', 'green', 'cadetblue', 'yellow',
                  'cyan', 'indianred', 'chartreuse', 'seagreen',
                  'darkorange', 'purple', 'aliceblue', 'olivedrab',
                  'deeppink', 'tan', 'rosybrown', 'khaki',
                  'aquamarine', 'cornflowerblue', 'saddlebrown',
                  'lightgray']

        self.noise_colors_ = []

        weighted_phase_err = np.sum(np.multiply(self.ips_, self.centers_array_[:, 7])) / np.sum(self.ips_)
        weighted_clust_err = np.sum(np.multiply(self.ips_, self.centers_array_[:, 8])) / np.sum(self.ips_)

        for i in range(self.n_comps_found_):
            if self.centers_array_[i, 7] > 5 * weighted_phase_err or \
                self.centers_array_[i, 8] > 5 * weighted_clust_err or \
                    self.ips_[i] <= 10:
                self.noise_colors_.append(colors[self.unique_labels_[i]])

    @abstractmethod
    def fit_over_one_dimensional_histograms(self, fig: object, axs,
                                            data_frame_object: object):
        """Fit over the histograms generated with the data frame object.

        Given a data frame object that has already been used
        to generate histograms for each dimension of data, this
        method will graph a GMM fit over each dimension. The returned
        matplotlib.plyplot figure may be shown with the method plt.show()
        or saved with the method plt.savefig() separately.

        Parameters
        ----------
        fig : matplotlib.pyplot figure
            The overarching figure object.

        axs : matplotlib.pyplot axes
            The object containing the four different histograms.

        data_frame_object : DataFrame class object
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.

        Returns
        -------
        fig : matplotlib.pyplot figure
            The overarching figure object.
        """
        pass

    @abstractmethod
    def get_pdf_fig(self, data_frame_object: object):
        """Plot the pdf of the Gaussian mixture on a surface.

        The returned matplotlib.plyplot figure can be shown and saved
        separately.

        Parameters
        ----------
        data_frame_object : DataFrame class object
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.

        Returns
        -------
        fig : matplotlib.pyplot figure
            The figure containing the pdf.

        save_string : str
            The recommended file name to use when saving the plot,
            which is done separately.
        """
        pass

    @abstractmethod
    def get_results_fig(self, data_frame_object: object):
        """Return the clustering results.

        The returned matplotlib.plyplot figure may be shown
        with the method plt.show() or saved with the method
        plt.savefig() separately. This method was written by
        Dwaipayan Ray and Adrian Valverde.

        Parameters
        ----------
        data_frame_object : object from the class DataFrame
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.

        Returns
        -------
        fig : matplotlib.pyplot figure
            The figure containing the clustered results.

        save_string : str
            The recommended file name to use when saving the plot,
            which is done separately.
        """
        pass

    def _set_gui(self, data_frame_object: object):
        """Set up the GUI to be used in the cluster merger method.

        Parameters
        ----------
        data_frame_object : object from the class DataFrame
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.

        Returns
        -------
        fig : matplotlib figure object
            The figure containing the clustered results.

        canvas : matplotlib canvas object
            The canvas that contains the drawing for the GUI.

        np_image : array-like, shape (n-rows, n-columns, 3)
            The image in np RGB array format.

        PIL_image : PIL image object
            The image in PIL format.

        color_list : list
            An empty list that will hold the colors to be merged.
        """
        plt.close()
        
        fig, _ = self.get_results_fig(data_frame_object=data_frame_object)

        canvas = FigureCanvas(fig)  # Initialize the canvas, which is the renderer that works with RGB values
        canvas.draw()  # Draw the canvas and cache the renderer
        ncols, nrows = fig.canvas.get_width_height()
        np_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((nrows, ncols, 3))
        # Convert figure to np_array, which is also an OpenCV image
        PIL_image = Image.fromarray(np_image).convert('RGB')  # Convert np array image to PIL image

        color_list = []  # Initialize cluster list

        return fig, canvas, np_image, PIL_image, color_list

    def cluster_merger(self, data_frame_object: object):
        """Merge clusters into one spot in the case of an over-fitting error.

        After showing the initial results of a clustering fit, this
        module allows the user to select multiple clusters to merge
        into a single cluster by clicking on them. Once the clusters
        are selected, the fit attributes 'n_comps_found_', 'labels_',
        'unique_labels_', 'colors_', and 'ips_', 'centers_array_', and
        'noise_colors_' are updated. The process
        repeats until all spots have been addressed. Before closing,
        it creates a final matplotlib figure of the results and returns
        it along with a save_string.

        Parameters
        ----------
        data_frame_object : object from the class DataFrame
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.

        Returns
        -------
        fig : matplotlib figure object
            The figure containing the clustered results.

        save_string : str
            The recommended file name to use when saving the plot,
            which is done separately.
        """
        if not self.clustered_:
            raise NotImplementedError("Must run a method to cluster the "
                                      "data before visualization.")

        fig, canvas, np_image, PIL_image, color_list = self._set_gui(data_frame_object=data_frame_object)

        done = False

        while not done:
            print("Look at the image and decide if there are any clusters you "
                  "want to merge. Once decided, close the window.")
            plt.show()
            merge = input("Are there clusters you want to merge? [y/n]")
            if merge == 'y':
                root = Tk()  # Initialize Tk
                PIL_Tk_image = ImageTk.PhotoImage(PIL_image)  # Create image that tkinter can use
                merger = ClusterMergerGUI(root, color_list=color_list, shape=np_image.shape, image=PIL_Tk_image,
                                          np_image=np_image)  # Initialize GUI
                print("Click on only the clusters you want to merge into one spot. If there are \n"
                      "multiple spots you want to do this with, close out the figure after addressing \n"
                      "the first spot and look for the next prompt.")
                root.mainloop()  # Run GUI

                brk = False
                if len(tuple(color_list)) > 1:
                    colors = ['blue', 'salmon', 'green', 'cadetblue', 'yellow',
                              'cyan', 'indianred', 'chartreuse', 'seagreen',
                              'darkorange', 'purple', 'aliceblue', 'olivedrab',
                              'deeppink', 'tan', 'rosybrown', 'khaki',
                              'aquamarine', 'cornflowerblue', 'saddlebrown',
                              'lightgray']

                    # Find indexes of colors
                    true_index_list = []  # Corresponds to the indices of the clusters in centers_array
                    cluster_index_list = []  # Corresponds to the indices of the clusters relative to labels_ and colors
                    for color in color_list:
                        try:
                            true_index_list.append(self.colors_.index(color))
                            cluster_index_list.append(colors.index(color))
                        except:
                            print("One of the colors selected, %s, doesn't match any cluster colors. "
                                  "Please try again." % color)
                            brk = True

                    if not brk:
                        ips_list = list(self.ips_)
                        max_ips = max(self.ips_[true_index_list])
                        true_index_keep = ips_list.index(max_ips)
                        cluster_index_keep = cluster_index_list[true_index_list.index(true_index_keep)]
                        other_true_indices = [i for i in true_index_list if i != true_index_keep]
                        other_true_indices.sort(reverse=True)
                        other_cluster_indices = [i for i in cluster_index_list if i != cluster_index_keep]
                        other_cluster_indices.sort(reverse=True)

                        # Adjust fit results accordingly
                        keeper_indices = list(range(self.n_comps_found_))
                        for i in other_true_indices:
                            keeper_indices.remove(i)
                        self.centers_array_ = self.centers_array_[keeper_indices, :]
                        self.n_comps_found_ -= len(other_true_indices)
                        for i in other_true_indices:
                            self.colors_.remove(self.colors_[i])
                        for i in other_cluster_indices:
                            self.labels_ = np.where(self.labels_ == i, cluster_index_keep, self.labels_)
                        self.unique_labels_ = np.unique(self.labels_)
                        labels_list = list(self.labels_)
                        ips = []
                        for n in self.unique_labels_:
                            cluster_ions = labels_list.count(n)
                            ips.append(cluster_ions)
                        self.ips_ = np.array(ips).reshape(-1,)

                        merged_cluster_true_index = \
                            true_index_keep - \
                            sum([1 if other_true_indices[x] < true_index_keep
                                 else 0 for x in range(len(other_true_indices))])

                        # Recalculate centers, but only for the merged clusters
                        self.recalculate_centers_uncertainties(data_frame_object=data_frame_object,
                                                               indices=merged_cluster_true_index)

                        if self.n_comps_found_ == 1:
                            break
                else:
                    if not brk:
                        print("Please select at least 2 clusters. If there are no more merges you want "
                              "to perform, enter 'n' when prompted.")

                fig, canvas, np_image, PIL_image, color_list = self._set_gui(
                    data_frame_object=data_frame_object)

            elif merge == 'n':
                done = True
            else:
                print("Invalid response. Please enter either 'y' or 'n'.")

                fig, canvas, np_image, PIL_image, color_list = self._set_gui(
                    data_frame_object=data_frame_object)

        fig, save_string = self.get_results_fig(data_frame_object=data_frame_object)

        return fig, save_string

    def export_to_clipboard(self):
        """Copies the centers, uncertainties, ips, and colors to
        an Excel format on the clipboard.
        """
        # Put all the columns of 'self.centers_array' into
        # proper formatting by rounding them according to
        # their uncertainties.
        if not self.clustered_:
            raise NotImplementedError("Must run a method to cluster the "
                                      "data before exporting to Excel.")

        rxs, rxs_err = sig_fig_rounder(
            self.centers_array_[:, 0].tolist(),
            self.centers_array_[:, 1].tolist())

        rys, rys_err = sig_fig_rounder(
            self.centers_array_[:, 2].tolist(),
            self.centers_array_[:, 3].tolist())

        rrs, rrs_err = sig_fig_rounder(
            self.centers_array_[:, 4].tolist(),
            self.centers_array_[:, 5].tolist())

        rps, rps_err = sig_fig_rounder(
            self.centers_array_[:, 6].tolist(),
            self.centers_array_[:, 7].tolist())

        clusters = []
        for n in range(len(rxs)):
            full_x = rxs[n] + '(' + rxs_err[n] + ')'
            # Convert the values to a proper format
            full_y = rys[n] + '(' + rys_err[n] + ')'
            full_r = rrs[n] + '(' + rrs_err[n] + ')'
            full_p = rps[n] + '(' + rps_err[n] + ')'
            cluster = [self.colors_[n], self.ips_[n][0], full_x,
                       full_y, full_r, full_p]
            # Combine the values into a list
            clusters.append(cluster)
            # Append the list to the clusters list

            export_frame = pd.DataFrame(
                clusters, columns=[
                    'Color', 'Shots', 'X', 'Y', 'R', 'P'])
            # Convert the clusters list to an Excel data frame
            export_frame = export_frame.sort_values(
                'Shots', axis=0, ascending=False)
            export_frame.to_clipboard(
                index=False, header=False)
            #  Export the data frame to the clipboard
