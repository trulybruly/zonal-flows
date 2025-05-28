import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import fftpack
import pandas as pd


# The effective area of the each coil is 3.8e-3 m^2. 

class MirnovCoils:
    def __init__(self, probe_ns, probe_files, time_file, tor_file, tor_coeff, zero_level_range=(0, 150)):
        self.probe_ns = probe_ns  # dict: {probe_name: area}
        self.probe_files = probe_files  # dict: {probe_name: csv_path}
        self.time_file = time_file  # path to time csv
        self.tor_file = tor_file  # path to toroidal field csv
        self.tor_coeff = tor_coeff
        self.zero_level_range = zero_level_range
        self.data = {}
        self.results = {}
        self._load_data()
        self._calculate_all()

    def _load_data(self):
        self.data['time'] = pd.read_csv(self.time_file, header=None).squeeze().values
        for name, file in self.probe_files.items():
            self.data[name] = pd.read_csv(file, header=None).squeeze().values
        self.data['tor'] = pd.read_csv(self.tor_file, header=None).squeeze().values

    def integrate(self, time, signal):
        dt = (time[-1] - time[0]) / (len(time) - 1)
        return np.cumsum(signal) * dt

    def zero_level(self, U, start, end):
        return U - np.mean(U[start:end])

    def smooth(self, U, window_length=25, polyorder=3):
        U_smooth = signal.savgol_filter(U, window_length=window_length, polyorder=polyorder, mode="nearest")
        U_smooth = U_smooth - np.mean(U_smooth[0:100])
        return U_smooth

    def _calculate_all(self):
        t = self.data['time']
        tor = self.data['tor']
        tor_int = self.integrate(t, tor * self.tor_coeff)
        self.results['tor_int'] = tor_int
        for name in self.probe_files:
            raw = self.data[name]
            zeroed = self.zero_level(raw, *self.zero_level_range)
            smoothed = self.smooth(zeroed)
            integrated = self.integrate(t, smoothed) / self.probe_ns[name]
            self.results[f'{name}_raw'] = raw
            self.results[f'{name}_zeroed'] = zeroed
            self.results[f'{name}_smoothed'] = smoothed
            self.results[f'{name}_integrated'] = integrated

    def plot_probe(self, name):
        plt.plot(self.data['time'], self.results[f'{name}_integrated'])
        plt.title(f"Integrated signal for {name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Signal (a.u.)")
        plt.show()

    def get_result(self, key):
        return self.results.get(key)

# Example usage:
# probe_ns = {'MP1': 150e-4, 'MP2': 199e-4, 'MP3': 181e-4, 'MP4': 182e-4}
# probe_files = {'MP1': 'MP1.csv', 'MP2': 'MP2.csv', ...}
# coils = MirnovCoils(probe_ns, probe_files, 'time.csv', 'tor.csv', 467.262162093)
# coils.plot_probe('MP1')
# result = coil.get_result('MP1_integrated')

