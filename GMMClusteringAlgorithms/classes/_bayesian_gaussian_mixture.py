"""Gaussian Mixture Model"""

# Author: Colin Weber
# Contact: colin.weber.27@gmail.com
# Contributors: Adrian Valverde, Dwaipayan Ray
# License: MIT
# sklearn License: BSD 3 clause


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from sklearn.mixture import BayesianGaussianMixture as BGM

from ..sklearn_mixture_piicr import StrictBayesianGaussianMixture
from ._data_frame import num_bins
from ._data_frame import shift_phase_dimension
from ._base import GaussianMixtureBase
from ._gaussian_mixture_model import wt_avg_unc_number
from ._gaussian_mixture_model import gauss_model_2save


class BayesianGaussianMixture(GaussianMixtureBase):
    """The class for implementing what we colloquially call the 'BGM' algorithms.

    Functionality includes: Cartesian or Polar Coordinates,
    Spherical, Tied, Diagonal, or Full covariance matrices,
    and general or strict fits.

    version : 0.1

    Parameters
    ----------
    n_components : int, defaults to 1.
        The maximum number of mixture components to use. If doing a
        strict fit, this is the number of components that will be
        used.


    cov_type : {'full' (default), 'tied', 'diag', 'spherical'}
        String describing the type of covariance parameters to use. Must be one of:
            'full'
                each component has its own general covariance matrix
            'tied'
                all components share the same general covariance matrix
            'diag'
                each component has its own diagonal covariance matrix
            'spherical'
                each component has its own single variance
        Taken from the sklearn package.

    tol : float, defaults to 1e-5
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold. Taken from
        the sklearn package.

    max_iter : int, defaults to 500.
        The number of EM iterations to perform. Taken from the sklearn package.

    n_init : int, defaults to 30.
        The number of initializations to perform. The best results
        are kept. Taken from the sklearn package.

    coordinates: {'Cartesian' (default), 'Polar'}
        The coordinate system to work in. Must be one of:
            'Cartesian'
            'Polar'

    weight_concentration_prior_type : str, defaults to 'dirichlet_process'.
        String describing the type of the weight concentration prior.
        Must be one of:
            'dirichlet_process' (using the Stick-breaking representation),
            'dirichlet_distribution' (can favor more uniform weights).
        Taken directly from sklearn package.

    weight_concentration_prior : float | None, optional.
        The dirichlet concentration of each component on the weight
        distribution (Dirichlet). This is commonly called gamma in the
        literature. The higher concentration puts more mass in
        the center and will lead to more components being active, while a lower
        concentration parameter will lead to more mass at the edge of the
        mixture weights simplex. The value of the parameter must be greater
        than 0. If it is None, it's set to ``1. / n_components``. Taken directly
        from sklearn package.

    Attributes
    ----------
    weights_ : array-like, shape (n_components,)
        The weights of each mixture components. Taken from the sklearn package.

    means_ : array-like, shape (n_components, n_features)
        The mean of each mixture component. Taken from the sklearn package.

    covariances_ : array-like
        The covariance of each mixture component.
        The shape depends on `covariance_type`:
            (n_components,)                        if 'spherical',
            (n_features, n_features)               if 'tied',
            (n_components, n_features)             if 'diag',
            (n_components, n_features, n_features) if 'full'

        Taken from the sklearn package.

    labels_ : array-like, shape (n_samples,)
        The label of each sample telling what component that sample
        belongs to.

    responsibilities_ : array-like, shape (n_samples, n_components)
        The probabilities that each sample belongs to each
        component.

    clustered_ : Boolean, defaults to False
        If True, then the Gaussian Mixture Model object has
        clustered the data.

    n_comps_found_ : int, defaults to n_components
        The number of components found by the fit.
        
    centers_array_ : array-like, shape (n_components, 9)
        An array containing the centers of each component and 
        their uncertainties in each of the 4 coordinate dimensions,
        as well as the cluster uncertainty.
        
    ips_ : array-like, shape (n_components,)
        The number of ions in each cluster.
        
    unique_labels_ : array-like, shape (n_components,)
        The labels used in the clustering fit.
        
    colors_ : list, len = n_components
        A list of the colors of each cluster.
    """

    def __init__(self, n_components=1, *, cov_type='full', tol=1e-5,
                 max_iter=500, n_init=30, coordinates='Cartesian',
                 weight_concentration_prior_type='dirichlet_process',
                 weight_concentration_prior=None):
        super().__init__(
            n_components=n_components, cov_type=cov_type, tol=tol,
            max_iter=max_iter, n_init=n_init)

        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.labels_ = None
        self.responsibilities_ = None
        self.coordinates = coordinates
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.clustered_ = False
        self.n_comps_found_ = self.n_components

    def _check_parameters(self):
        """Check the parameters that don't originate in the base model."""
        if not (isinstance(self.coordinates, str)):
            raise TypeError("The parameter 'coordinates' must be a "
                            "string, but got type %s instead." %
                            type(self.coordinates))

        if not (self.coordinates in ['Cartesian', 'Polar']):
            raise ValueError("The parameter 'coordinates' must be "
                             "either 'Cartesian' or 'Polar', but"
                             "got %s instead." % self.coordinates)

    def _BGM_fit(self, x):
        """Fit a Bayesian Gaussian Mixture to the data given by x.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_attributes)
            The data to be fit.

        Returns
        -------
        model : BayesianGaussianMixture from the sklearn package
            The BayesianGaussianMixture object that has been fit to the data.
        """
        model = BGM(n_components=self.n_components, tol=self.tol,
                    max_iter=self.max_iter, n_init=self.n_init,
                    covariance_type=self.cov_type,
                    weight_concentration_prior_type=self.weight_concentration_prior_type,
                    weight_concentration_prior=self.weight_concentration_prior)
        data = x.astype('float32')
        model.fit(data)

        return model

    def _strict_BGM_fit(self, x):
        """Fit a Strict Bayesian Gaussian Mixture to the data given by x.

        A Strict Bayesian Gaussian Mixture is a Bayesian Gaussian Mixture
        in which the number of components is required to remain at the
        value given by 'n_components'. The fitting algorithm accomplishes
        this by returning the model parameters as soon as the number of
        samples in the smallest cluster gets below 0.1% of all the samples
        in the data set. Therefore, this model rarely converges. In this way,
        it is a special case of the regular Bayesian Gaussian Mixture,
        where the number of components is allowed to be lower than 'n_components'.

        Parameters
        x : array-like, shape (n_samples, n_attributes)
            The data to be fit.

        Returns
        -------
        model : StrictBayesianGaussianMixture from the sklearn package
            The StrictBayesianGaussianMixture object that has been fit to the data.
        """
        model = StrictBayesianGaussianMixture(
            n_components=self.n_components, tol=self.tol,
            max_iter=self.max_iter, n_init=self.n_init,
            covariance_type=self.cov_type,
            weight_concentration_prior_type=self.weight_concentration_prior_type,
            weight_concentration_prior=self.weight_concentration_prior,
            min_cluster_size=0.001)
        data = x.astype('float32')
        model.strict_fit(data)

        return model

    def _calc_secondary_centers_unc(self, c1s, c1s_err, c2s,
                                    c2s_err, data_frame_object):
        """Calculate the coordinates of the cluster centers for the coordinate system that
        was not used for the fit.

        Standard errors are calculated with typical standard error propagation methods.
        Initialize the attribute 'centers_array_'.

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
        xC, yC = data_frame_object.center
        xC_unc, yC_unc = data_frame_object.center_unc

        if self.coordinates == 'Cartesian':
            # Calculate the radii and phases of cluster centers, with standard errors
            rs = np.sqrt(np.square(np.subtract(c1s, xC)) +
                         np.square(np.subtract(c2s, yC)))
            rs_err = np.sqrt(np.add(np.add(np.multiply(np.square(np.divide(
                np.subtract(c1s, xC), rs)), np.square(c1s_err)), np.multiply(
                np.square(np.divide(np.subtract(c1s, xC), rs)), np.square(xC_unc))), np.add(
                np.multiply(np.square(np.divide(np.subtract(c2s, yC), rs)),
                            np.square(c2s_err)), np.multiply(
                    np.square(np.divide(np.subtract(c2s, yC), rs)), np.square(yC_unc)))))
            ps = np.rad2deg(np.arctan(np.divide(np.subtract(c2s, yC), np.subtract(c1s, xC))))
            for i in range(len(c1s)):
                if c1s[i] - xC < 0:
                    ps[i] += 180
                if c1s[i] - xC > 0 > c2s[i] - yC:
                    ps[i] += 360
            ps_err = np.rad2deg(np.sqrt(np.add(np.add(np.square(np.multiply(np.multiply(
                np.divide(1, np.add(1, np.square(np.divide(np.subtract(c2s, yC),
                                                           np.subtract(c1s, xC))))),
                np.divide(np.subtract(yC, c2s), np.square(np.subtract(c1s, xC)))),
                c1s_err)), np.square(np.multiply(np.multiply(np.divide(1, np.add(
                    1, np.square(np.divide(np.subtract(c2s, yC), np.subtract(c1s, xC))))),
                                                             np.divide(np.subtract(
                                                                 yC, c2s), np.square(
                                                                 np.subtract(c1s, xC)))), xC_unc))), np.add(
                np.square(np.multiply(np.multiply(np.divide(1, np.add(1, np.square(
                    np.divide(np.subtract(c2s, yC), np.subtract(c1s, xC))))),
                                                  np.divide(1, np.subtract(c1s, xC))),
                                      c2s_err)), np.square(np.multiply(np.multiply(
                                        np.divide(1, np.add(1, np.square(np.divide(np.subtract(c2s, yC),
                                                            np.subtract(c1s, xC))))),
                                        np.divide(1, np.subtract(c1s, xC))), yC_unc))))))

            cluster_err = np.sqrt(np.add(np.square(c1s_err),
                                         np.square(c2s_err)))

            self.centers_array_ = np.vstack((c1s, c1s_err, c2s,
                                             c2s_err, rs, rs_err,
                                             ps, ps_err,
                                             cluster_err)).T

        else:  # if self.coordinates == 'Polar':
            # Calculate the x- and y- coordinates of the cluster centers, with standard errors.
            # First, ensure data is not phase shifted.
            if data_frame_object.phase_shifted_:
                shift = data_frame_object.phase_shift_

                c2s += shift
                c2s = np.where(c2s > 360, c2s - 360, c2s)

                self.means_[:, 1] += shift
                self.means_[:, 1] = np.where(
                    self.means_[:, 1] > 360, self.means_[:, 1] - 360, self.means_[:, 1])

                p_raw = data_frame_object.data_array_[:, 3]
                p_raw += shift
                p_raw = np.where(p_raw > 360, p_raw - 360, p_raw)
                data_frame_object.data_array_[:, 3] = p_raw
                data_frame_object.phase_shifted_ = False

            # Convert to radians
            phases = np.deg2rad(c2s)
            phases_err = np.deg2rad(c2s_err)

            xs = np.add(np.multiply(c1s, np.cos(phases)), xC)
            xs_err = np.sqrt(np.add(np.add(np.square(np.multiply(np.cos(
                phases), c1s_err)), np.square(np.multiply(
                    np.multiply(c1s, np.sin(phases)), phases_err))),
                xC_unc ** 2))
            ys = np.add(np.multiply(c1s, np.sin(phases)), yC)
            ys_err = np.sqrt(np.add(np.add(np.square(np.multiply(np.sin(
                phases), c1s_err)), np.square(np.multiply(
                    np.multiply(c1s, np.cos(phases)), phases_err))),
                yC_unc ** 2))
            cluster_err = np.sqrt(np.add(np.square(xs_err),
                                         np.square(ys_err)))

            self.centers_array_ = np.vstack((xs, xs_err, ys, ys_err,
                                             c1s, c1s_err, c2s,
                                             c2s_err, cluster_err)).T

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
        self.unique_labels_ = np.unique(self.labels_)
        self.n_comps_found_ = len(self.unique_labels_)
        labels_list = self.labels_.tolist()
        ips = []
        for n in self.unique_labels_:
            cluster_ions = labels_list.count(n)
            ips.append(cluster_ions)
        self.ips_ = np.array(ips).reshape(-1, 1)

        self.means_ = self.means_[self.unique_labels_, :]
        self.weights_ = self.weights_[self.unique_labels_]

        c1s = self.means_[:, 0]
        c2s = self.means_[:, 1]

        if self.cov_type == 'spherical':
            self.covariances_ = self.covariances_[self.unique_labels_]
            c1s_err = np.sqrt(self.covariances_ / np.array(ips))
            c2s_err = c1s_err
        elif self.cov_type == 'diag':
            self.covariances_ = self.covariances_[self.unique_labels_, :]
            c1s_err = np.sqrt(
                self.covariances_[:, 0] / np.array(ips))
            c2s_err = np.sqrt(
                self.covariances_[:, 1] / np.array(ips))
        elif self.cov_type == 'tied':
            c1_covs = np.full(
                (self.n_comps_found_,), self.covariances_[0, 0])
            c2_covs = np.full(
                (self.n_comps_found_,), self.covariances_[1, 1])
            c1s_err = np.sqrt(c1_covs / np.array(ips))
            c2s_err = np.sqrt(c2_covs / np.array(ips))
        else:  # if self.cov_type == 'full':
            self.covariances_ = self.covariances_[self.unique_labels_, :, :]
            c1s_err = np.sqrt(
                self.covariances_[:, 0, 0] / np.array(ips))
            c2s_err = np.sqrt(
                self.covariances_[:, 1, 1] / np.array(ips))

        self._calc_secondary_centers_unc(c1s, c1s_err, c2s, c2s_err,
                                         data_frame_object)

        colors = ['blue', 'salmon', 'green', 'cadetblue', 'yellow',
                  'cyan', 'indianred', 'chartreuse', 'seagreen',
                  'darkorange', 'purple', 'aliceblue', 'olivedrab',
                  'deeppink', 'tan', 'rosybrown', 'khaki',
                  'aquamarine', 'cornflowerblue', 'saddlebrown',
                  'lightgray']

        self.colors_ = []
        for i in self.unique_labels_:
            self.colors_.append(colors[i])

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
        if not self.clustered_:
            raise NotImplementedError("Must run a method to cluster the "
                                      "data before recalculating centers.")

        if not data_frame_object.phase_shifted_:
            shift_phase_dimension(data_frame_object)

        cp = data_frame_object.data_array_

        cluster_ind = np.arange(0, self.n_comps_found_)  # array of cluster numbers

        c1s, c1s_err, c1_chi_sq, c1_red_chi_sq, c1_sigma_abs, \
            c1_sigma_err, c1_height_abs, c1_height_err, c1_fw_hm_abs, \
            c1_fw_hm_err = [], [], [], [], [], [], [], [], [], []
        c2s, c2s_err, c2_chi_sq, c2_red_chi_sq, c2_sigma_abs, \
            c2_sigma_err, c2_height_abs, c2_height_err, c2_fw_hm_abs, \
            c2_fw_hm_err = [], [], [], [], [], [], [], [], [], []
        cluster_err = []

        for i in cluster_ind:
            plt.figure()
            c1_cut = []
            c2_cut = []
            if self.coordinates == 'Cartesian':
                for j in range(len(self.labels_)):
                    if self.labels_[j] == self.unique_labels_[i]:
                        c1_cut.append(cp[:, 0][j])
                        c2_cut.append(cp[:, 1][j])
            else:  # if self.coordinates == 'Polar'
                for j in range(len(self.labels_)):
                    if self.labels_[j] == self.unique_labels_[i]:
                        c1_cut.append(cp[:, 2][j])
                        c2_cut.append(cp[:, 3][j])

            width_c1 = max(c1_cut) - min(c1_cut)
            width_c2 = max(c2_cut) - min(c2_cut)

            if len(c1_cut) == 1 or len(c2_cut) == 1:
                c1s.append(c1_cut[0])
                c1s_err.append(0)
                c1_chi_sq.append(0)
                c1_red_chi_sq.append(0)
                c1_sigma_abs.append(0)
                c1_sigma_err.append(0)
                c1_height_abs.append(0)
                c1_height_err.append(0)
                c1_fw_hm_abs.append(0)
                c1_fw_hm_err.append(0)

                c2s.append(c2_cut[0])
                c2s_err.append(0)
                c2_chi_sq.append(0)
                c2_red_chi_sq.append(0)
                c2_sigma_abs.append(0)
                c2_sigma_err.append(0)
                c2_height_abs.append(0)
                c2_height_err.append(0)
                c2_fw_hm_abs.append(0)
                c2_fw_hm_err.append(0)

                cluster_err.append(0)

            else:
                c1_fit = gauss_model_2save(
                    min(c1_cut) - 0.1 * width_c1,
                    max(c1_cut) + 0.1 * width_c1, c1_cut,
                    num_bins(c1_cut), 'cadetblue')
                c2_fit = gauss_model_2save(
                    min(c2_cut) - 0.1 * width_c2,
                    max(c2_cut) + 0.1 * width_c2, c2_cut,
                    num_bins(c2_cut), 'darkorange')

                c1_fit_array = np.array(c1_fit)
                c2_fit_array = np.array(c2_fit)
                if np.isnan(c1_fit_array.astype(float)[1]) \
                        or np.isnan(c2_fit_array.astype(float)[1]) \
                        or 0.0001 >= c1_fit_array[1] or \
                        0.0001 >= c2_fit_array[1]:

                    c1, c1_err = wt_avg_unc_number(c1_cut, width_c1)
                    c2, c2_err = wt_avg_unc_number(c2_cut, width_c2)

                    c1s.append(c1)
                    c1s_err.append(c1_err)
                    c1_chi_sq.append(0)
                    c1_red_chi_sq.append(0)
                    c1_sigma_abs.append(0)
                    c1_sigma_err.append(0)
                    c1_height_abs.append(0)
                    c1_height_err.append(0)
                    c1_fw_hm_abs.append(0)
                    c1_fw_hm_err.append(0)

                    c2s.append(c2)
                    c2s_err.append(c2_err)
                    c2_chi_sq.append(0)
                    c2_red_chi_sq.append(0)
                    c2_sigma_abs.append(0)
                    c2_sigma_err.append(0)
                    c2_height_abs.append(0)
                    c2_height_err.append(0)
                    c2_fw_hm_abs.append(0)
                    c2_fw_hm_err.append(0)

                    cluster_err.append(
                        np.sqrt(c2_err ** 2 + c1_err ** 2))

                else:
                    c1s.append(c1_fit_array[0])
                    c1s_err.append(c1_fit_array[1])
                    c1_chi_sq.append(c1_fit_array[2])
                    c1_red_chi_sq.append(c1_fit_array[3])
                    c1_sigma_abs.append(c1_fit_array[4])
                    c1_sigma_err.append(c1_fit_array[5])
                    c1_height_abs.append(c1_fit_array[6])
                    c1_height_err.append(c1_fit_array[7])
                    c1_fw_hm_abs.append(c1_fit_array[8])
                    c1_fw_hm_err.append(c1_fit_array[9])

                    c2s.append(c2_fit_array[0])
                    c2s_err.append(c2_fit_array[1])
                    c2_chi_sq.append(c2_fit_array[2])
                    c2_red_chi_sq.append(c2_fit_array[3])
                    c2_sigma_abs.append(c2_fit_array[4])
                    c2_sigma_err.append(c2_fit_array[5])
                    c2_height_abs.append(c2_fit_array[6])
                    c2_height_err.append(c2_fit_array[7])
                    c2_fw_hm_abs.append(c2_fit_array[8])
                    c2_fw_hm_err.append(c2_fit_array[9])

                    cluster_err.append(
                        np.sqrt(c1_fit_array[1] ** 2 +
                                c2_fit_array[1] ** 2))

            plt.title('BGM %i comps c1_bins (cadetblue) = %i ; c2_bins '
                      '(orange) = %i\n(c1, c2) = (%0.2f,%0.2f),'
                      'Cluster unc=%0.5f' % (
                        self.n_comps_found_, num_bins(c1_cut),
                        num_bins(c2_cut), c1s[i], c2s[i], cluster_err[i]))
            plt.xlim(-10, 10)
            plt.show()

        self._calc_secondary_centers_unc(c1s, c1s_err, c2s, c2s_err,
                                         data_frame_object)

    def _cluster_data_one_d(self, x):
        """Use the Bayesian Gaussian Mixture fit from the sklearn package to cluster the data.

        Assigns the object the attributes 'means_',
        'covariances_', 'weights_', 'labels_', 'responsibilities_',
        and 'n_comps_found_'.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_attributes)
            The data to be clustered.
        """
        model = self._BGM_fit(x)

        # Assign attributes
        self.means_ = model.means_
        self.covariances_ = model.covariances_
        self.weights_ = model.weights_
        self.labels_ = model.predict(x)
        self.responsibilities_ = model.predict_proba(x)
        self.n_comps_found_ = np.shape(self.weights_)[0]

    def cluster_data(self, data_frame_object):
        """Use the Bayesian Gaussian Mixture from the sklearn package to cluster the data.

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
        self._check_base_parameters()

        # Gather data from data_frame_object
        if self.coordinates == 'Cartesian':
            data = data_frame_object.data_array_[
                   :, (0, 1)]
        else:  # if self.coordinates == 'Polar':
            if not data_frame_object.phase_shifted_:
                shift_phase_dimension(data_frame_object)
            data = data_frame_object.data_array_[
                   :, (2, 3)]

        model = self._BGM_fit(data)  # Fit model to data

        # Assign attributes
        self.labels_ = model.predict(data)
        self.means_ = model.means_
        self.covariances_ = model.covariances_
        self.weights_ = model.weights_
        self.responsibilities_ = model.predict_proba(data)

        self.clustered_ = True

        self._calculate_centers_uncertainties(data_frame_object)

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
        self._check_base_parameters()

        # Gather data from data_frame_object
        if self.coordinates == 'Cartesian':
            data = data_frame_object.data_array_[
                   :, (0, 1)]
        else:  # if self.coordinates == 'Polar':
            if not data_frame_object.phase_shifted_:
                shift_phase_dimension(data_frame_object)
            data = data_frame_object.data_array_[
                   :, (2, 3)]

        model = self._strict_BGM_fit(data)

        # Assign attributes
        self.means_ = model.means_
        self.covariances_ = model.covariances_
        self.weights_ = model.weights_
        self.labels_ = model.predict(data)
        self.responsibilities_ = model.predict_proba(data)
        self.n_comps_found_ = np.shape(self.weights_)[0]
        self.clustered_ = True

        self._calculate_centers_uncertainties(data_frame_object)

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
        self._check_base_parameters()

        if not data_frame_object.phase_shifted_:
            shift_phase_dimension(data_frame_object)
        data = data_frame_object.data_array_

        for n in range(0, np.shape(data)[1]):
            raw_data = data[:, n].reshape(-1, 1)
            data_range = max(raw_data) - min(raw_data)
            self._cluster_data_one_d(raw_data)
            x_values_min = min(raw_data) - 0.1 * data_range
            x_values_max = max(raw_data) + 0.1 * data_range
            x_values = np.linspace(x_values_min, x_values_max, 1000)
            pdf_values = np.array([0] * 1000).reshape(-1, 1)
            for i in range(0, self.n_comps_found_):
                mean = self.means_[i]
                sigma = np.sqrt(self.covariances_[i])
                y_values = scipy.stats.norm.pdf(
                    x_values, loc=mean, scale=sigma).reshape(-1, 1)
                y_values *= self.weights_[i]
                pdf_values = np.add(pdf_values, y_values)
            pdf_values *= len(data) * (x_values_max - x_values_min) / num_bins(raw_data)

            dim_binary = str(format(n, '02b'))
            row = int(dim_binary[0])
            col = int(dim_binary[1])
            axs[row, col].plot(x_values, pdf_values)

        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.labels_ = None
        self.responsibilities_ = None
        self.n_comps_found_ = self.n_components

        return fig

    def plot_pdf_surface(self, data_frame_object):
        """Plot the pdf of the Bayesian Gaussian mixture on a surface.

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
        if not self.clustered_:
            raise NotImplementedError("Must run a method to cluster the "
                                      "data before visualization.")

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        if self.coordinates == 'Cartesian':
            x1 = data_frame_object.data_array_[:, 0]
            x2 = data_frame_object.data_array_[:, 1]

            x1_range = max(x1) - min(x1)
            x1_grid_min = min(x1) - 0.1 * x1_range
            x1_grid_max = max(x1) + 0.1 * x1_range
            x1_grid_len = np.complex(0, 1000)

            x2_range = max(x2) - min(x2)
            x2_grid_min = min(x2) - 0.1 * x2_range
            x2_grid_max = max(x2) + 0.1 * x2_range
            x2_grid_len = np.complex(0, 1000)

        else:  # if self.coordinates == 'Polar':
            x1 = data_frame_object.data_array_[:, 2]
            x1_range = max(x1) - min(x1)
            x1_grid_min = min(x1) - 0.1 * x1_range
            x1_grid_max = max(x1) + 0.1 * x1_range
            x1_grid_len = np.complex(0, 1000)

            x2_grid_min = 0
            x2_grid_max = 360
            x2_grid_len = np.complex(0, 1000)

        x1_values, x2_values = np.mgrid[
                               x1_grid_min:x1_grid_max:x1_grid_len,
                               x2_grid_min:x2_grid_max:x2_grid_len]
        # Generate meshgrid of x1, x2 values to plot versus the
        # pdf of the GMM fit.
        grid = np.dstack((x1_values, x2_values))
        # Organize into a format usable by the .pdf function
        gmm_values = np.zeros((1000, 1000))
        # Initialize a data array that will hold the values of
        # the BGM for the corresponding x values

        for n in range(0, self.n_comps_found_):
            mean = self.means_[n]
            # Grab both dimensions of the mean from the model
            # result
            if self.cov_type == 'tied':
                cov = self.covariances_
            else:
                cov = self.covariances_[n]
                # Grab the covariance matrix from the model result.
            z_values = scipy.stats.multivariate_normal.pdf(
                grid, mean=mean, cov=cov)
            # Given a mean and covariance matrix, this calculates the pdf of a multivariate
            # Gaussian function for each x,y value.
            z_values *= self.weights_[n]
            # Weight the output by the weights given by the trained model
            gmm_values = np.add(gmm_values, z_values)
            # Add this array element-wise to the GMM array

        if self.coordinates == 'Polar':
            rads = x1_values
            phases = np.deg2rad(x2_values)
            x1_values = np.multiply(rads, np.cos(phases))
            x2_values = np.multiply(rads, np.sin(phases))

        ax.plot_surface(x1_values, x2_values, gmm_values,
                        cmap='viridis')
        title_string = 'BGM PDF (Cov type=%s, ' \
                       'n-components=%i)\n' % (self.cov_type,
                                               self.n_comps_found_)
        title_string += data_frame_object.file[0:-4] + '\n'
        title_string += 'TOF cut=(%.3f,%.3f), ' % \
                        data_frame_object.tof_cut
        title_string += 'Ion cut=%s, ' % (data_frame_object.ion_cut,)
        title_string += 'Rad cut=(%.1f,%.1f)' % \
                        data_frame_object.rad_cut
        title_string += 'Time cut=%s' % (data_frame_object.time_cut,)
        plt.title(title_string)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Probability Density')

        save_string = 'BGM %s PDF, %s, %s, %s Clusters, ' \
                      'timecut=%s,radcut=%s,tofcut=%s,' \
                      'ioncut=%s.jpeg' % (
                          self.coordinates, data_frame_object.file[0:-4],
                          self.cov_type, self.n_comps_found_,
                          data_frame_object.time_cut, data_frame_object.rad_cut,
                          data_frame_object.tof_cut, data_frame_object.ion_cut)

        return fig, save_string

    def show_results(self, data_frame_object):
        """Display the clustering results.

        The returned matplotlib.plyplot figure may be shown
        with the method plt.show() or saved with the method
        plt.savefig() separately. This method was written by
        Dwaipayan Ray and Adrian Valverde.

        Parameters
        ----------
        data_frame_object : DataFrame class object
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
        if not self.clustered_:
            raise NotImplementedError("Must run a method to cluster the "
                                      "data before visualization.")

        data_array = data_frame_object.data_array_
        n_samples = len(data_array[:, 0])
        center_array = self.centers_array_

        fig = plt.figure()
        ax = plt.subplot(111, aspect='equal')
        axs = plt.gca()

        plt.xlim(-14, 8)
        plt.ylim(-7, 13)

        plt.title("BGM %s total counts: %i; total clusters: %i, "
                  "Cov=%s\n%s\nTOF cut=%s, Ion cut=%s, Rad cut=%s, "
                  "Time cut=%s" % (self.coordinates,
                                   n_samples, self.n_comps_found_,
                                   self.cov_type,
                                   data_frame_object.file[0:-4],
                                   data_frame_object.tof_cut,
                                   data_frame_object.ion_cut,
                                   data_frame_object.rad_cut,
                                   data_frame_object.time_cut))

        plt.xlabel('X [mm]', weight='bold')
        plt.ylabel('Y [mm]', weight='bold')

        discarded_counts_stuff = '%i counts (%.1f%%)' % (
            (len(data_array[:, 0]) - sum(self.ips_)),
            100.0 * (n_samples - sum(self.ips_)) / n_samples)

        labels = []
        for i in range(len(self.ips_)):
            labels.append(
                r"%i counts (%.1f%%), x=%.3f$\pm$ %.3f, "
                r"y=%.3f$\pm$%.3f, r=%.3f$\pm$%.3f, "
                r"p=%.3f$\pm$%.3f" % (
                    self.ips_[i], 100.0 * self.ips_[i] / n_samples,
                    self.centers_array_[i, 0],
                    self.centers_array_[i, 1],
                    self.centers_array_[i, 2],
                    self.centers_array_[i, 3],
                    self.centers_array_[i, 4],
                    self.centers_array_[i, 5],
                    self.centers_array_[i, 6],
                    self.centers_array_[i, 7]))
        labels.append(discarded_counts_stuff)

        label_indices = np.arange(0, len(self.unique_labels_))
        for k, col, index in zip(self.unique_labels_, self.colors_,
                                 label_indices):
            my = self.labels_ == k
            axs.add_artist(
                plt.Circle((center_array[index, 0],
                            center_array[index, 2]),
                           radius=center_array[index, 8],
                           fill=False, color='red'))

            plt.plot(data_array[my, 0], data_array[my, 1], 'o',
                     color=col, markersize=2, label="%s:\n%s" % (
                    col, labels[index]))

        plt.errorbar(center_array[:, 0], center_array[:, 2],
                     yerr=center_array[:, 3], xerr=center_array[:, 1],
                     elinewidth=1, capsize=1, ls='', marker='o',
                     markersize=1, color='red')

        legend_stuff = plt.legend(
            loc='center left', bbox_to_anchor=(1, 0.5), numpoints=1,
            fontsize=10, labelspacing=0.5)
        for j in range(len(legend_stuff.legendHandles)):
            legend_stuff.legendHandles[j]._legmarker.set_markersize(10)

        plt.grid()

        save_string = 'BGM %s PDF, %s, %s, %s Clusters, ' \
                      'timecut=%s,radcut=%s,tofcut=%s,' \
                      'ioncut=%s.jpeg' % (
                          self.coordinates, data_frame_object.file[0:-4],
                          self.cov_type, self.n_comps_found_,
                          data_frame_object.time_cut, data_frame_object.rad_cut,
                          data_frame_object.tof_cut, data_frame_object.ion_cut)

        return fig, save_string
