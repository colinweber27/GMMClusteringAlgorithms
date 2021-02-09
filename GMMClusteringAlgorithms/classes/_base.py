"""Base class for specific mixture models both included in
sklearn and developed using the sklearn algorithms."""

# Author: Colin Weber
# Contact: colin.weber.27@gmail.com
# License: MIT
# sklearn License: BSD 3 clause

from abc import ABCMeta, abstractmethod

import pandas as pd
import numpy as np


def sig_fig_rounder(data: list, unc: list):
    """Rounds the data based on its uncertainty according to
    standard significant figure rules.

    Parameters
    ----------
    data : list, len(n_samples)
        The data to be rounded.

    unc : list, len(n_samples)
        The uncertainty in the data.
    """
    for u in range(len(unc)):
        if unc[u] <= 0:
            unc[u] = 0.001
        unc_string = str(unc[u])
        unc_string = unc_string.replace('0', '')
        unc_string = unc_string.replace('.', '')
        data_string = str(data[u])
        data_string = data_string.replace('0', '')
        data_string = data_string.replace('.', '')
        if data_string[-1] == '5':
            data_string = str(data[u]) + '1'
            data[u] = float(data_string)
        if unc_string[0] != '1':
            if unc_string[-1] == '5':
                unc_string = str(unc[u]) + '1'
                unc[u] = float(unc_string)
            decimal_places = -int(np.floor(np.log10(unc[u])))
            data[u] = str(round(data[u], decimal_places))
            unc[u] = str(round(unc[u], decimal_places))
            if len(data[u]) < abs(decimal_places) + 2 and \
                    data[u][0] != '-':
                while len(data[u]) != abs(decimal_places) + 2:
                    data[u] += '0'
            if len(data[u]) < abs(decimal_places) + 3 and \
                    data[u][0] == '-':
                while len(data[u]) != abs(decimal_places) + 3:
                    data[u] += '0'
            if len(unc[u]) < abs(decimal_places) + 2 and \
                    unc[u][-1] != '2':
                while len(unc[u]) != abs(decimal_places) + 2:
                    unc[u] += '0'
            if float(unc[u]) >= 2:
                data[u] = str(int(round(float(data[u]), 0)))
                unc[u] = str(int(round(float(unc[u]), 0)))
        else:
            if unc_string[-1] == '5':
                unc_string = str(unc[u]) + '1'
                unc[u] = float(unc_string)
            decimal_places = 1 - int(np.floor(np.log10(unc[u])))
            data[u] = str(round(data[u], decimal_places))
            unc[u] = str(round(unc[u], decimal_places))
            test_unc = unc[u].replace('0', '')
            test_unc = test_unc.replace('.', '')
            if test_unc[0] == '2':
                decimal_places -= 1
            if len(data[u]) < abs(decimal_places) + 2 and \
                    data[u][0] != '-':
                while len(data[u]) != abs(decimal_places) + 2:
                    data[u] += '0'
            if len(data[u]) < abs(decimal_places) + 3 and \
                    data[u][0] == '-':
                while len(data[u]) != abs(decimal_places) + 3:
                    data[u] += '0'
            if len(unc[u]) < abs(decimal_places) + 2 and \
                    unc[u][-1] != '2':
                while len(unc[u]) != abs(decimal_places) + 2:
                    unc[u] += '0'
            if float(unc[u]) >= 2:
                data[u] = str(int(round(float(data[u]), 0)))
                unc[u] = str(int(round(float(unc[u]), 0)))
    return data, unc


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
    def fit_over_one_dimensional_histograms(self, fig, axs,
                                            data_frame_object):
        """Given a data frame object that has already been used
        to generate histograms for each dimension of data, this
        method will graph a GMM fit over each dimension.

        Parameters
        ----------
        fig : matplotlib.pyplot figure
            The overarching figure object.

        axs : matplotlib.pyplot axes
            Contains the four different histograms.

        data_frame_object : object from the class DataFrame
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.
        """
        pass

    @abstractmethod
    def cluster_data(self, data_frame_object):
        """Cluster the data.

        Assigns the mixture object the attributes 'means_',
        'covariances_', 'weights_', 'labels_', 'responsibilities_',
        and 'n_comps_found'.

        Parameters
        ----------
        data_frame_object : object from the class DataFrame
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.
        """
        pass

    @abstractmethod
    def cluster_data_strict(self, data_frame_object):
        """Cluster the data, but restricts n_components to the value
        of the parameter 'n_components'.

        Assigns the mixture object the attributes 'means_',
        'covariances_', 'weights_', 'labels_', 'responsibilities_',
        and 'n_comps_found'.

        Parameters
        ----------
        data_frame_object : object from the class DataFrame
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.
        """
        pass

    @abstractmethod
    def _cluster_data_one_d(self, x):
        """Cluster the data, which in this case must have only 1
        dimension.

        Assigns the mixture object the attributes 'means_',
        'covariances_', 'weights_', 'labels_', 'responsibilities_',
        and 'n_comps_found'.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_attributes)
            The data to be clustered
        """
        pass

    @abstractmethod
    def _calculate_centers_uncertainties(self, data_frame_object):
        """After clustering the data, organize and return the
         centers and uncertainties in the centers of each cluster.
         This is done to facilitate plotting the results and
         exporting them to an Excel data frame.

        Parameters
        ----------
        data_frame_object : object from the class DataFrame
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.
        """
        pass

    @abstractmethod
    def recalculate_centers_uncertainties(self, data_frame_object):
        """A method that recalculates the centers of each cluster
        and the uncertainties in the centers.

        It is a different method from simply extracting the centers
        and uncertainties from the fit. Instead, it does a Gaussian
        fit to each dimension of each cluster and uses the
        statistics from the fits to calculate the centers.

        Parameters
        ----------
        data_frame_object : object from the class DataFrame
            The object that contains the processed data and
            information that is used by the algorithms to do the
            fits.
        """
        pass

    @abstractmethod
    def plot_pdf_surface(self, data_frame_object):
        """Plot the pdf of the Gaussian mixture on a surface.

        Parameters
        ----------
        data_frame_object : object from the class DataFrame
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
        """Display the clustering results.

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

    def export_to_excel(self):
        """Copies the centers, uncertainties, ips, and colors to
        an Excel format on the clipboard.
        """
        # Put all the columns of 'self.centers_array' into
        # proper formatting by rounding them according to
        # their uncertainties.
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