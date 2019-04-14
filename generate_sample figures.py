import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from process_voyager_images import *

# find pattern of two peaks and return the index where it ends
def find_peaks(signal, i, window_size=250):
    data = signal[i: i + window_size]
    data_deriv = np.zeros_like(data)
    data_deriv[0:-2] = data[1:-1]
    data_deriv = data-data_deriv
    data_deriv_abs = np.abs(data_deriv)
    data_deriv_filt = filter_signal(data_deriv_abs, filter_size=21)
    data_deriv_step = (data_deriv_filt>0.008).astype(np.int)
    
    return data, data_deriv, data_deriv_abs, data_deriv_filt, data_deriv_step


# plot signal helper
def plot(line, xlabel, ylabel, save_path=None, is_new_figure=True):
    if is_new_figure:
        plt.figure()
    plt.plot(range(line.size), line)
    if np.max(line) > 0.2:
        plt.ylim([-0.2, 1.1])
    else:
        plt.ylim([-0.2, 0.2])
        
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save_path != None:
        plt.savefig(save_path)
#        plt.close()
    else:
        plt.show()

if __name__=='__main__':
    plt.close('all')
    start_idx = 5999900 + 6350
    data1, data_deriv, data_deriv_abs, data_deriv_filt, data_deriv_step = \
    find_peaks(data, start_idx, 4500*1)
    plot(data1, 'Time steps', 'Amplitude', 'images/initial_signal.png')
    
    start_idx = 5999900 + 6650
    data2, data_deriv, data_deriv_abs, data_deriv_filt, data_deriv_step = \
    find_peaks(data, start_idx, 250)
    plot(data2[0:-10], 'Time steps', 'Amplitude', 'images/small_signal.png')
    
    plot(data_deriv[0:-10], 'Time steps', 'Amplitude', 'images/small_signal_deriv.png')
    
    plot(data_deriv_abs[0:-10], 'Time steps', 'Amplitude', 'images/small_signal_deriv_abs.png')
    
    plot(data_deriv_filt[0:-10], 'Time steps', 'Amplitude', 'images/small_signal_deriv_filt.png')
    
    plot(data_deriv_step[0:-10], 'Time steps', 'Amplitude', 'images/small_signal_deriv_step.png')
    
    