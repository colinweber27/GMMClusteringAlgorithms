"""Data frame class for converting .lmf files for use with
Gaussian Mixture Models from the sklearn package."""

# Author: Colin Weber
# Contact: colin.weber.27@gmail.com
# Contributors: Adrian Valverde and Dwaipayan Ray
# License: MIT

import os
import struct
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def num_bins(x):
    """Calculate the number of bins to use for a histogram of the data x.

    Uses the common rule of thumb num_bins = sqrt(x).

        Parameters
        ----------
        x : array-like, shape (n_samples, )
            The data to be binned.

        Returns
        -------
        n : int
            The number of bins to use
    """
    n = int(np.ceil(np.sqrt(len(x))))
    return n


def _check_directory(directory):
    """Check the parameter 'directory' to make sure it is an existing str.

    Parameters
    ----------
    directory: str
        The directory to check.
    """
    if not isinstance(directory, str):
        raise TypeError(
            "The parameter 'directory' must be a string, but got "
            "type %s instead." % type(directory))

    if not os.path.isdir(directory):
        raise ValueError(
            "The parameter 'directory' specifies a directory which "
            "does not exist. Either create a new directory, or "
            "change the parameter to point to an existing one.")


def shift_phase_dimension(data_frame_object: object):
    """Shift the phase dimension of the data prior to analyzing.

    This is done to minimize the errors that arise because of
    Python's not recognizing that 0deg = 360deg.

    Parameters
    ----------
    data_frame_object : object from the class DataFrame
        The object that contains the processed data and
        information that is used by the algorithms to do the
        fits.
    """
    if not data_frame_object.processed_:
        print("Must run method 'process_lmf' before running "
              "'shift_phase_dimension'.")
        exit()

    phase_data = data_frame_object.data_array_[:, 3]
    p_raw_bins = plt.hist(
        phase_data, bins=num_bins(phase_data), range=(0, 360),
        weights=None, alpha=0.6)  # Bin the data
    centers = (p_raw_bins[1][:-1] + p_raw_bins[1][1:]) / 2.0
    # Extract the centers
    p_raw_mask = p_raw_bins[0] == min(p_raw_bins[0])
    # Generate a matrix of all False except True at the index of
    # the minimum bin.
    shift = centers[p_raw_mask][0]
    # Get the center of the minimum bin
    phase_data = phase_data - shift
    # Shift all the data points down such that the minimum bin
    # is now centered near 0deg.
    phase_data = np.where(
        phase_data < 0, phase_data + 360, phase_data)
    # For all the points lower than 0deg, add 360deg to put them
    # in the correct range.
    plt.close()
    # Because we don't need to see the histogram of the data.
    data_frame_object.data_array_[:, 3] = phase_data
    data_frame_object.phase_shift_ = shift

    data_frame_object.phase_shifted_ = True


class DataFrame:
    """Class for the data frame.

    This class is used as an organizational tool for the lmf files
    and their parameters. Objects from this class are what is
    manipulated by the clustering algorithms.

    version : 0.1

    Parameters
    ----------
    directory: str, defaults to active directory
        The directory that contains the .lmf file to be clustered.

    file: str, defaults to None
        The .lmf file to be clustered. Requires entire filename,
        including extension '.lmf'.

    center: tuple, defaults to (0,0)
        The Cartesian coordinates of the center of the CPT.

    rad_cut: tuple, defaults to (0, np.inf)
        The limits of the radial dimension. Only ion hits within
        the limits, inclusive, will be analyzed.

    ion_cut: tuple, defaults to (0, np.inf)
        The limits on the number of ions present in each data point.
        If the number of ions in a point falls outside of these
        values, that point will not be analyzed.

    tof_cut: tuple, defaults to (-50000,-10000)
        The limits on the tof of the points, in ns. If a point's
        tof falls outside this range, it will not be analyzed.

    time_cut: tuple, defaults to (0, np.inf)
        The limits on the time duration of the entire experiment,
        in seconds. 0 corresponds to the beginning of the run.
        Only samples with a timestamp within this window will be
        analyzed.

    Attributes
    ----------
    data_array_ : array-like, shape (n_samples, 4)
        The data array containing the processed data from the .lmf
        file. The 0th and 1st columns represent the data in
        Cartesian coordinates, and the 2nd and 3rd columns represent
        the data in polar coordinates.

    data_frame_ : Pandas Data Frame
        The data frame containing the processed data, but in Excel
        format (.xlsx).

    processed_ : Boolean
        True if the data in the frame has been processed, false
        otherwise.

    phase_shifted_ : Boolean
        True if the phase dimension has been shifted to allow for
        better fitting, false otherwise.

    phase_shift_ : float in range[0, 360)
        The amount the phase dimension has been shifted prior to
        fitting. This is done to improve the fit results by
        minimizing errors that result from Python's not naturally
        recognizing that 0deg = 360deg.
    """

    def __init__(self, *, directory=os.path.dirname(os.path.realpath(__file__)),
                 file=None, center=(0, 0), center_unc=(0, 0), rad_cut=(0, np.inf),
                 ion_cut=(0, np.inf), tof_cut=(-50000, -10000), time_cut=(0, np.inf)):
        self.directory = directory
        self.file = file
        self.center = center
        self.center_unc = center_unc
        self.rad_cut = rad_cut
        self.ion_cut = ion_cut
        self.tof_cut = tof_cut
        self.time_cut = time_cut
        self.processed_ = False
        self.phase_shifted_ = False
        self.data_frame_ = None
        self.data_array_ = None
        self.phase_shift_ = 0

    def _check_initial_parameters(self):
        """Check values of the inputted parameters."""
        _check_directory(self.directory)

        if not isinstance(self.file, str):
            raise TypeError("The parameter 'file' "
                            "must be a string, but got "
                            "type %s instead." %
                            type(self.directory))

        if not self.file.lower().endswith('.lmf'):
            raise ValueError("The parameter 'file' needs to be"
                             "a string of a valid .lmf file with"
                             "the string '.lmf' at the end.")

        if not isinstance(self.center, tuple):
            raise TypeError("The parameter 'center' "
                            "should be a tuple, but got "
                            "type %s instead." %
                            type(self.center))

        if not isinstance(self.center_unc, tuple):
            raise TypeError("The parameter 'center_unc' "
                            "should be a tuple, but got "
                            "type %s instead." %
                            type(self.center))

        if not isinstance(self.rad_cut, tuple):
            raise TypeError("The parameter 'rad_cut' "
                            "should be a tuple, but got "
                            "type %s instead." %
                            type(self.rad_cut))

        if self.rad_cut[0] >= self.rad_cut[1]:
            raise ValueError("The parameter 'rad_cut' "
                             "should be a tuple listing"
                             "first the lower limit, and"
                             "then the upper limit, of the"
                             "radial dimension.")

        if not isinstance(self.ion_cut, tuple):
            raise TypeError("The parameter 'ion_cut' "
                            "should be a tuple, but got "
                            "type %s instead." %
                            type(self.ion_cut))

        if self.ion_cut[0] >= self.ion_cut[1]:
            raise ValueError("The parameter 'ion_cut' "
                             "should be a tuple listing"
                             "first the lower limit, and"
                             "then the upper limit, of the"
                             "number of ions.")

        if not isinstance(self.tof_cut, tuple):
            raise TypeError("The parameter 'tof_cut' "
                            "should be a tuple, but got "
                            "type %s instead." %
                            type(self.tof_cut))

        if self.tof_cut[0] >= self.tof_cut[1]:
            raise ValueError("The parameter 'tof_cut' "
                             "should be a tuple listing"
                             "first the lower limit, and"
                             "then the upper limit, of the"
                             "allowable tof values for the data.")

        if not isinstance(self.time_cut, tuple):
            raise TypeError("The parameter 'time_cut' "
                            "should be a tuple, but got "
                            "type %s instead." %
                            type(self.time_cut))

        if self.time_cut[0] >= self.time_cut[1]:
            raise ValueError("The parameter 'time-cut' "
                             "should be a tuple listing"
                             "first the lower limit, and"
                             "then the upper limit, of the"
                             "allowable time stamps for the data.")

    def process_lmf(self):
        """Process the .lmf data file.

        This method was written by Dwaipayan Ray and
        Adrian Valverde.

        Returns
        --------
        lmf_start_time_int : int
            The start time of the data run.

        lmf_stop_time_int : int
            The end time of the data run.
        """
        self._check_initial_parameters()

        kx = 1.29
        ky = 1.31

        xC = self.center[0]
        yC = self.center[1]

        filename = self.directory + '/' + self.file

        # Read the .lmf file
        file = open(filename, 'rb')  # 'rb' r for read, b for binary
        file.seek(0, 2)  # Go to the last byte of the file '2'=os.SEEK_END
        LMFileSize = file.tell()  # Returns the position of the last byte = file's size
        file.seek(0, 0)  # Go to the beginning of the file '0'=os.SEEK_BEG
        LMHeadVersion = (struct.unpack('<I', file.read(4)))[0]
        # Read 4 bytes and converts them to unsigned int
        LMDataFormat = struct.unpack('<I', file.read(4))[0]
        # Read 4 bytes and converts them to unsigned int
        LM64NumberOfCoordinates = struct.unpack('<Q', file.read(8))[0]
        # Read 8 bytes and converts them to unsigned int
        LM64HeaderSize = struct.unpack('<Q', file.read(8))[0]
        # Read 8 bytes and converts them to unsigned int
        LM64UserHeaderSize = struct.unpack('<Q', file.read(8))[0]
        # Read 8 bytes and converts them to unsigned int
        LM64NumberOfEvents = struct.unpack('<Q', file.read(8))[0]
        # Read 8 bytes and converts them to unsigned int
        file.seek(4, 1)  # skip 4B '1'=os.SEEK_CUR
        LMStartTime = struct.unpack('<Q', file.read(8))[0]
        # Read 8 bytes and converts them to unsigned int
        file.seek(4, 1)  # skip 4B '1'=os.SEEK_CUR
        LMStopTime = struct.unpack('<Q', file.read(8))[0]
        # Read 8 bytes and converts them to unsigned int

        # Read the data block
        dataBlockOffset = LM64HeaderSize + LM64UserHeaderSize
        dataBlockSize = LMFileSize - dataBlockOffset
        file.seek(dataBlockOffset, 0)
        dataBlock = file.read(dataBlockSize)

        file.close()

        ''' Conversion of timeStamp strings to integers
        LMStartTime_int and LMStopTime_int are both in seconds 
        from 18:00:00 1969-12-31 in your local time'''
        lmf_start_time_int = int(LMStartTime)
        lmf_stop_time_int = int(LMStopTime)
        start_time = time.strftime(
            '%Y-%m-%d %H-%M-%S', (time.localtime(LMStartTime)))  # type(start_time) = str
        stop_time = time.strftime(
            '%Y-%m-%d %H-%M-%S', (time.localtime(LMStopTime)))  # type(stop_time_ = str
        runtime = lmf_stop_time_int - lmf_start_time_int

        print("%s" % self.file)
        print("Start:           %s" % start_time)
        print("Stop:            %s" % stop_time)
        print("Runtime [s]:     %d" % runtime)

        dataOffset = 0  # absolute position in the data_block

        eventList = []
        ''' event_list stores events. Only events with 1, 2, 3, 4 
        and 7 non-zero channels are stored. This is to remove events
         where one of the triggers didn't fire for any number of 
         reasons.'''

        # Loop through all events in the file
        for event in range(0, LM64NumberOfEvents):
            eventSize = dataBlock[dataOffset: dataOffset + 4]
            eventSize = struct.unpack('<I', eventSize)[0]
            dataOffset += 4

            eventTrigger = dataBlock[dataOffset: dataOffset + 4]
            eventTrigger = struct.unpack('<I', eventTrigger)[0]
            dataOffset += 4

            eventNumber = dataBlock[dataOffset: dataOffset + 8]
            eventNumber = struct.unpack('<Q', eventNumber)[0]
            dataOffset += 8

            eventTimeStamp = dataBlock[dataOffset: dataOffset + 8]
            eventTimeStamp = struct.unpack('<Q', eventTimeStamp)[0]
            dataOffset += 8

            # eventTimeStamp is counted from the first event
            if event == 0:
                eventTimeStampOffset = eventTimeStamp

            eventTimeStamp -= eventTimeStampOffset

            eventID = 0x0000
            TDC = [0] * 8
            # Loop through all 8 channels
            for channel in range(0, 8):
                channelStatus = 0
                channelValue = 0

                channelStatus = dataBlock[dataOffset: dataOffset + 2]
                channelStatus = struct.unpack('<H', channelStatus)[0]
                dataOffset += 2

                if channelStatus != 0:
                    channelValue = dataBlock[dataOffset: dataOffset + 4]
                    channelValue = struct.unpack('<i', channelValue)[0]
                    dataOffset += 4

                TDC[channel] = channelValue
                eventID <<= 1
                eventID = eventID | channelStatus

            # Skip reading Clock and Trigger channels of TDC
            dataOffset += 2 * (2 + 4)

            # If an event has non-zero channels 1, 2, 3, 4 and 7, then it is appended to the eventList
            if (eventID & 0x00F2) == 0x00F2:
                x1 = TDC[0] * 0.001
                x2 = TDC[1] * 0.001
                y1 = TDC[2] * 0.001
                y2 = TDC[3] * 0.001
                mcp = TDC[6] * 0.001
                x = 0.5 * kx * (x1 - x2)
                y = 0.5 * ky * (y1 - y2)
                radius = np.sqrt((x - xC) ** 2 + (y - yC) ** 2)
                phase_deg = np.degrees(np.arctan2((y - yC), (x - xC)))
                if phase_deg < 0:
                    phase_deg = phase_deg + 360

                sum_x = (x1 + x2)
                sum_y = (y1 + y2)
                diff_xy = sum_x - sum_y
                tof = TDC[6] * 0.001
                timeStamp = eventTimeStamp * 1.0e-12

                trig = eventTimeStamp * 1.0e-12
                eventList.append(
                    [x, y, tof, timeStamp, radius, phase_deg, x1, x2, y1, y2, sum_x, sum_y, diff_xy, mcp, trig])
            # END Loop through all events in the file

        ''' Conversion of eventList to DataFrame '''
        data_df_prel = pd.DataFrame(
            eventList, columns=[
                'X', 'Y', 'Tof', 'TStamp', 'Radius',
                'Phase', 'X1', 'X2', 'Y1', 'Y2',
                'SumX', 'SumY', 'DiffXY', 'MCP', 'trig'])
        data_df_prel.loc[:, 'trig'] = data_df_prel['trig'].round(decimals=2)
        
        '''This dataframe has every data point from every event 
        which triggered all 5 TDC channels.'''

        # Impose limits :
        # sum_x, sum_y and diff_xy to get rid of noise
        # ion_cut for ion-ion-interaction
        # tof_cut and rad_cut to clean up the spectra a little bit

        # If an event is too far away from the circle, or
        # there were too many ions, or it took too long
        # to get to the PS-MCP from the trap, it's also cut from the
        # data set.
        data_df_prel = data_df_prel.query('45<SumX<48')
        data_df_prel = data_df_prel.query('44<SumY<47')

        data_df_prel = data_df_prel.reset_index().set_index("trig")
        data_df_prel["Ions_that_shot"] = \
            data_df_prel.reset_index().groupby("trig").trig.count()

        data_df_prel = data_df_prel.query(
            '%i<Ions_that_shot<%i' % (
                self.ion_cut[0], self.ion_cut[1]))  # ion_cut
        data_df_prel = data_df_prel.query(
            '%i<Tof<%i' % (
                self.tof_cut[0], self.tof_cut[1]))  # tof_cut
        data_df_prel = data_df_prel.query(
            '%f<Radius<%f' % (
                self.rad_cut[0], self.rad_cut[1]))  # rad_cut
        data_df_prel = data_df_prel.query(
            '%f<TStamp<%f' % (
                self.time_cut[0], self.time_cut[1]))  # timestamp cut

        # Final DataFrame
        data_df = pd.DataFrame([])
        data_df['X [mm]'] = data_df_prel['X']
        data_df['Y [mm]'] = data_df_prel['Y']
        data_df['Tof [ns]'] = data_df_prel['Tof']
        data_df['TStamp [s]'] = data_df_prel['TStamp']
        data_df['Radius [mm]'] = data_df_prel['Radius']
        data_df['Phase [deg]'] = data_df_prel['Phase']
        data_df['Ions in that Shot'] = data_df_prel["Ions_that_shot"]

        data_df['TDC Ch1: X1 [ns]'] = data_df_prel['X1']
        data_df['TDC Ch2: X2 [ns]'] = data_df_prel['X2']
        data_df['TDC Ch3: Y1 [ns]'] = data_df_prel['Y1']
        data_df['TDC Ch4: Y2 [ns]'] = data_df_prel['Y2']
        data_df['TDC Ch7: MCP [ns]'] = data_df_prel['MCP']
        data_df['SumX [ns]'] = data_df_prel['SumX']
        data_df['SumY [ns]'] = data_df_prel['SumY']
        data_df['DiffXY [ns]'] = data_df_prel['DiffXY']

        self.processed_ = True
        self.data_frame_ = data_df

        self.data_array_ = data_df.values[:, (0, 1, 4, 5)]

        return lmf_start_time_int, lmf_stop_time_int

    def return_processed_data_excel(self):
        """Return the processed data as an Excel file.

        The returned writer may be saved separately with the
        method writer.save(). This method was written by Dwaipayan Ray
        and Adrian Valverde.

        Returns
        -------
        writer : Pandas ExcelWriter object
            The Excel object containing the processed data.
        """
        if not self.processed_:
            raise NotImplementedError(
                "Must run method '_process_lmf' before running "
                "'_save_processed_data'.")

        writer = pd.ExcelWriter(
            self.directory + '%s, tof_cut=%s, ion_cut=%s, '
                             'rad_cut=%s, time_cut=%s.xlsx' %
            (self.file[0:-4], self.tof_cut, self.ion_cut,
             self.rad_cut, self.time_cut))
        self.data_frame_.to_excel(writer, 'sheet1')

        workbook = writer.book
        worksheet = writer.sheets['sheet1']
        format1 = workbook.add_format({'center_across': True})

        # Formatting and column width
        worksheet.set_column(
            0, len(self.data_frame_.columns) + 1, 20, format1)

        return writer

    def get_data_figure(self):
        """Plot the data and return the figure.

        The data figure will have to be shown after returning
        using the method plt.show(), and saved after retuning
        using the method plt.savefig(). The method returns a
        string that can be used as a file name for the figure.

        This method was written by Dwaipayan Ray and Adrian
        Valverde.

        Returns
        -------
        plot : matplotlib.pyplot figure
            The figure containing the axes.

        save_string : str
            A string that contains all the relevant information for
            saving, which is done separately.
        """
        if not self.processed_:
            raise NotImplementedError("Must run method '_process_lmf' "
                                      "before running '_show_figure'.")

        x_raw = self.data_array_[:, 0]
        y_raw = self.data_array_[:, 1]

        x_edges = np.arange(-20, 20, 0.25)
        y_edges = np.arange(-20, 20, 0.25)
        h, x_edges, y_edges = np.histogram2d(
            x_raw, y_raw, bins=(x_edges, y_edges))
        h = np.rot90(h)
        h = np.flipud(h)
        # 'rot90' and 'flipud' basically make a transpose
        ions = np.ma.masked_where(h == 0, h)
        # Matrix of "True" for all "o"-s, and "False" for everything
        # else. Then all "True"-s will be white and "False"-s based
        # on the number in them.

        '''Plot Spectra'''
        # Initialize figure
        plot = plt.figure(figsize=(6, 6))
        ax = plt.subplot(111, aspect='equal')

        # Change limits
        r_max = max(self.data_array_[:, 2])
        plt.xlim(self.center[0] - 1.2 * r_max, self.center[0] + 1.2 * r_max)
        plt.ylim(self.center[1] - 1.2 * r_max, self.center[1] + 1.2 * r_max)

        # Set axis labels
        plt.xlabel('X [mm]')
        plt.ylabel('Y [mm]')

        # Plot the data
        plt.pcolormesh(x_edges, y_edges, ions, cmap='viridis')
        c_bar = plt.colorbar()
        c_bar.set_label('Counts')

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        save_string = '%s_UNCLUSTERED,tof_cut=%s,ion_cut=%s,' \
                      'rad_cut=%s,time_cut=%s.jpeg' % \
                      (self.file[0:-4], self.tof_cut, self.ion_cut,
                       self.rad_cut, self.time_cut)

        return plot, save_string

    def get_one_dimension_histograms_plot(self):
        """Return a plot object consisting of four histograms.

        Each axis of the plot contains the histogram for one
        dimension of data. Also returns a string that can be
        used for saving the plot. The returned matplotlib
        figure can be shown with the method plt.show() and
        saved with the method plt.savefig() separately.

        Returns
        -------
        fig : matplotlib.pyplot figure object
            The overarching figure.

        axs : matplotlib.pyplot axis object
            The object containing the four different histograms.

        save_string : str
            The recommended file name to use when saving the plot,
            which is done separately.
        """
        if not self.phase_shifted_:
            shift_phase_dimension(self)

        # Calculate histogram range of each dimension
        widths = [max(self.data_array_[:, i]) -
                  min(self.data_array_[:, i]) for
                  i in range(0, np.shape(self.data_array_)[1])]

        fig, axs = plt.subplots(2, 2, sharey='all')  # Initialize figure

        # For each data dimension except phase, plot a histogram
        # Do this by converting each of the four dimensions to base 2
        # and using these representations to correspond to the 4
        # subplots.
        for dim in range(0, np.shape(self.data_array_)[1] - 1):
            dim_binary = str(format(dim, '02b'))
            row = int(dim_binary[0])
            col = int(dim_binary[1])
            # Plot the histograms
            axs[row, col].hist(
                self.data_array_[:, dim],
                bins=num_bins(self.data_array_[:, dim]),
                range=(min(self.data_array_[:, dim]) - 0.1 *
                       widths[dim], max(self.data_array_[:, dim]) +
                       0.1 * widths[dim]),
                weights=None, alpha=0.6, color='blue',
                histtype='stepfilled')
            axs[row, col].set(ylabel='Counts')
        # Plot the phase histogram
        axs[1, 1].hist(
            self.data_array_[:, 3],
            bins=num_bins(self.data_array_[:, 3]),
            range=(0, 360), weights=None, alpha=0.6, color='blue',
            histtype='stepfilled')
        axs[1, 1].set(ylabel='Counts')

        # Add axis labels
        axs[0, 0].set(title='X Dimension', xlabel='X(mm)')
        axs[0, 1].set(title='Y Dimension', xlabel='Y(mm)')
        axs[1, 0].set(title='Radial Dimension', xlabel='Radius(mm)')
        axs[1, 1].set(title='Phase Dimension', xlabel='Phase(deg)')

        # Finishing touches
        fig.suptitle(
            'Histograms of Dimensions\n%s\nTOF cut=%s, Ion cut=%s, '
            'Rad cut=%s, Time cut=%s' %
            (self.file[0:-4], self.tof_cut, self.ion_cut, self.rad_cut,
             self.time_cut))
        fig.tight_layout()

        save_string = 'Histograms of Dimensions, %s, tof_cut=%s, ' \
                      'ion_cut=%s, rad_cut=%s, time_cut=%s.jpeg' \
                      % (self.file[0:-4], self.tof_cut, self.ion_cut,
                         self.rad_cut, self.time_cut)

        return fig, axs, save_string
