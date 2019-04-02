import scipy.io.wavfile
import matplotlib.pyplot as plt
import numpy as np
import pdb
from PIL import Image

# load data and invert signals
in_wav_path = 'data/voyager_left_channel_32bitfloat_384kHz.wav'
rate, data = scipy.io.wavfile.read(in_wav_path)

# correct for waveforms
data = -data

# run things like TV playback - adds single line at a time
def add_image_line(image_data, line):
    scan_width, lines = image_data.shape
    
    # go and shift the scan lines to make space for new line
    for i in range(511):
        image_data[:, i] = image_data[:, i+1]

    # add new line
    image_data[:, -1] = line
    
    return image_data

# gaussian filter weights
def gaussian_weights(filter_size=5):
    f_idx = (filter_size-1)/2
    f_indices = np.arange(-f_idx, f_idx+1, 1)
    sigma = filter_size/5
    A = 1/(sigma * np.sqrt(2*np.pi))
    B = np.exp(-0.5 * (f_indices/sigma)**2 )
    
    return A * B

# simple filtering of signal to reduce noise/ anti-aliasing
def filter_signal(signal, filter_size=5):
    filter_weights = gaussian_weights(filter_size)
    
    filtered_signal = np.zeros_like(signal)
    padded_signal = np.zeros((signal.size + filter_size-1))
    fs = int((filter_size-1)/2)
    padded_signal[fs: fs + signal.size] = signal
    
    for i in np.arange(fs, padded_signal.size - fs):
        filtered_signal[i - fs] = np.sum(padded_signal[i - fs: i + fs + 1] * filter_weights)
    
    return filtered_signal

# two peaks checker - uses signal derivative
def check_for_two_peaks(data):
    indices = []
    match_val = 0
    matches = 0
    for i in range(data.size):
        if data[i] == match_val and matches != 4:
            indices.append(i)
            match_val -= 1
            match_val = np.abs(match_val)
            matches += 1
        
        if data[i] == match_val and matches == 4:
            # match found
            indices.append(i)
            return True, indices
        
    return False, indices

# find pattern of two peaks and return the index where it ends
def find_peaks(signal, start_idx, stride=50, window_size=250):
    for i in np.arange(start_idx, signal.size, stride):
        data = signal[i: i + window_size]
        data_deriv = np.zeros_like(data)
        data_deriv[0:-2] = data[1:-1]
        data_deriv = np.abs(data-data_deriv)
        data_deriv = filter_signal(data_deriv, filter_size=21)
        data_deriv = (data_deriv>0.008).astype(np.int)
        check, indices = check_for_two_peaks(data_deriv)
        if check:
            data_select = np.zeros_like(data)
            data_select[indices[1+2]:indices[2+2]] = 1
            data_select = data * data_select
            
            index = np.argmin(data_select)
            
            return index + i

# plot signal helper
def plot(line):
    plt.plot(range(line.size), line)
    plt.ylim([-0.4, 0.4])
    plt.show()

# renormalize to pixel values
def renormalize(line):
    min_val = -0.1#np.min(line)
    max_val = 0.1#np.max(line)
    line = np.clip(line, min_val, max_val)
    
    return ((line - min_val)/(max_val-min_val))*255

def bilinear_resize(data, out_shape):
    samples = data.size
    
    # if downsampling
    if out_shape < samples:
        # run a basic anti-aliasing filter
        data = filter_signal(data, filter_size=5)
        
    # create bilinnear sampling weights
    weights = np.linspace(0, samples-2, out_shape)
    
    # create container to store output
    out_data = np.zeros((out_shape))
    
    for i in range(out_shape):
        
        # fetch weights
        weight = weights[i]
        
        # get the last index with known value
        first_index = int(np.floor(weight))
        
        # get weight for last index
        first_weight = 1-(weight - first_index)
        
        # combine last and next index linearly
        value = data[first_index] * first_weight + data[first_index+1] * (1-first_weight)
        
        # save resampled values
        out_data[i] = value
    
    return out_data
        
def renormalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    
    return ((image - min_val)/(max_val-min_val))*255
    
if __name__=='__main__':
    start_idx = 5999900
    lines = 512
    height = 512
    start_idx = find_peaks(data, start_idx)
    image_data = np.zeros((height, lines))
    
    save_idx = 1
    for i in range(lines*107):
        end_idx = find_peaks(data, start_idx + 2700)
        line = data[start_idx : end_idx]
        start_idx = end_idx
        if line.size < 3500 and line.size > 2500:
            line_buffer = bilinear_resize(renormalize(line), height)
#            line_buffer = np.zeros((height))
#            line_buffer[0:line.size] = line
            image_data = add_image_line(image_data, line_buffer)
        else:
            print('Found bigger line: %d \nSkipping...' % line.size)
            if line.size < 2500:
                pdb.set_trace()
            
        if i % 128 == 0:
            buffer = 'output/%0.5d.png' % save_idx
            print('Scan: %d | Image: %d' % ( i % 512, int(i / 512) ))
            print('Saving image to: %s' % buffer)
            save_image = Image.fromarray(image_data.astype(np.uint8))
            save_image.save(buffer)
            save_idx += 1
