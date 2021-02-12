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
    def __init__(self, n_components, cov_type, tol, max_iter,
                 n_init):
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
                                    c2s_err, data_frame_object):
        """Calculate the coordinates of the cluster centers for the coordinate system
        that was not used for the fit.

        Standard errors are calculated with typical standard error propagation methods.

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
    def _calculate_centers_uncertainties(self, data_frame_object):
        """After clustering the data, organize the cluster centers into a more accessible format.

        Assigns the attributes 'centers_array_', 'ips_',
        'unique_labels', and 'colors_'.

        Parameters
        ----------
        data_frame_object : DataFrame class object
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.
        """
        pass

    @abstractmethod
    def recalculate_centers_uncertainties(self, data_frame_object):
        """Recalculate the centers of each cluster and the uncertainties in the centers.

        This uses a different method from simply extracting the centers
        and uncertainties from the fit. Instead, it fits a univariate Gaussian
        to each dimension of each cluster and uses the statistics from the
        fits to calculate the centers. This method was written by Dwaipayan
        Ray and Adrian Valverde.

        Parameters
        ----------
        data_frame_object : DataFrame class object
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.
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
    def cluster_data(self, data_frame_object):
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
    def cluster_data_strict(self, data_frame_object):
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

    @abstractmethod
    def fit_over_one_dimensional_histograms(self, fig, axs,
                                            data_frame_object):
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
            Contains the four different histograms.

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
    def plot_pdf_surface(self, data_frame_object):
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
    def show_results(self, data_frame_object):
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
            Contains the clustered results.

        save_string : str
            The recommended file name to use when saving the plot,
            which is done separately.
        """
        pass

    def cluster_merger(self, data_frame_object):
        """Merge clusters into one spot in the case of an over-fitting error.

        After showing the initial results of a clustering fit, this
        module allows the user to select multiple clusters to merge
        into a single cluster by clicking on them. Once the clusters
        are selected, the fit attributes 'n_comps_found_', 'labels_',
        'unique_labels_', 'colors_', and 'ips_' are updated. The process
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
            Contains the clustered results.

        save_string : str
            The recommended file name to use when saving the plot,
            which is done separately.
        """
        if not self.clustered_:
            raise NotImplementedError("Must run a method to cluster the "
                                      "data before visualization.")

        fig, save_string = self.show_results(data_frame_object=data_frame_object)
        
        canvas = FigureCanvas(fig)  # Initialize the canvas, which is the renderer that works with RGB values
        canvas.draw()  # Draw the canvas and cache the renderer
        ncols, nrows = fig.canvas.get_width_height()
        np_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((nrows, ncols, 3))
        # Convert figure to np_array, which is also an OpenCV image
        PIL_image = Image.fromarray(np_image).convert('RGB')  # Convert np array image to PIL image

        color_list = []  # Initialize cluster list

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
                    true_index_list.append(self.colors_.index(color))
                    cluster_index_list.append(colors.index(color))
                ips_list = list(self.ips_)
                true_index_keep = ips_list.index(max(self.ips_[true_index_list]))
                cluster_index_keep = cluster_index_list[true_index_list.index(true_index_keep)]
                other_true_indices = [i for i in true_index_list if i != true_index_keep]
                other_true_indices.sort(reverse=True)
                other_cluster_indices = [i for i in cluster_index_list if i != cluster_index_keep]
                other_cluster_indices.sort(reverse=True)

                # Adjust fit results accordingly
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
                self.ips_ = np.array(ips)

                # Recalculate centers
                self.recalculate_centers_uncertainties(data_frame_object=data_frame_object)

                if self.n_comps_found_ == 1:
                    break

                # Generate new figure
                fig, save_string = self.show_results(data_frame_object=data_frame_object)

                # Prepare environment for GUI
                canvas = FigureCanvas(fig)  # Initialize the canvas, which is the renderer that works with RGB values
                canvas.draw()  # Draw the canvas and cache the renderer
                ncols, nrows = fig.canvas.get_width_height()
                np_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((nrows, ncols, 3))
                # Convert figure to np_array, which is also an OpenCV image
                PIL_image = Image.fromarray(np_image).convert('RGB')  # Convert np array image to PIL image

                color_list = []  # Initialize cluster list
            elif merge == 'n':
                done = True
            else:
                print("Invalid response. Please enter either 'y' or 'n'.")

                fig, save_string = self.show_results(data_frame_object=data_frame_object)

                # Prepare environment for GUI
                canvas = FigureCanvas(fig)  # Initialize the canvas, which is the renderer that works with RGB values
                canvas.draw()  # Draw the canvas and cache the renderer
                ncols, nrows = fig.canvas.get_width_height()
                np_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((nrows, ncols, 3))
                # Convert figure to np_array, which is also an OpenCV image
                PIL_image = Image.fromarray(np_image).convert('RGB')  # Convert np array image to PIL image

                color_list = []  # Initialize cluster list

        fig, save_string = self.show_results(data_frame_object=data_frame_object)

        return fig, save_string

    def export_to_excel(self):
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
