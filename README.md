# VoyagerImageDecoder
Decoding images from Voyager Disc

![image](https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/imageGif.gif)


# Introduction
The year is 1977, NASA is attempting to send two space probes that entail what we as humans are capable of. It captures our ability to dream things that are much bigger than us and aims to outlast our existence (atleast for some of the visionary scientists who came up with the idea). The two missions are, appropriately, called Voyager I and II as they begin their voyage into the unknown,  attempting to reach the furthest any man-made probe has every been. Along with instruments for recording data and communicating back to earth, scientist have prepared a time capsule, showing our existence to any potential alien civilization that would find these probes. However, the technology at the time limits them to only be able to send audio records. They have devised a genius idea to send much more than audio.

Famously know as the golden records due to the gold plating, the voyager records contain audio including greetings in many different languages, along with music, sounds of animals and places on earth and a concise collection of images showing different aspects of human existence. The images are encoded in a way so that they could be written as audio clips into one of the two golden disks. 

<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/golden_disk.jpg?raw=true" alt="golden_disk.jpg" width="400"/>


# Instructions for aliens
As bizzare as it sounds, instructions were given to aliens in the form of a disk cover, showing in a very concise and descriptive way - the origins of the probe and what needed to be done to decode the images. In this post, I attempt to decode these images by observing the instructions as well as getting help from existing work on voyager golden disk image browser [^rf1]. 

<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/golden_disk_cover.jpg?raw=true" alt="golden_disk_cover.jpg" width="400"/>


## Clues for decoding images
The first clue appears in the top right of the cover where you can see a waveform:
<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/golden_disk_cover_waveform.jpg?raw=true" alt="golden_disk_cover_waveform.jpg" width="400"/>

As noted in [^rf1] the wave form appear to be sequentially written data with each complete waveform representing a time-series signal, while the whole signal is split into groups of multiple of these individual time-series signals. Notice the binary number below each peaks and valleys in the signal indicating that each one of them correspond to a value in the time-series, while the binary numbers above each waveform indicate that they are part of a bigger sequence. 

If you look close at the images just below the waveforms, you may be able to find clues on what each of the two numbers correspond to.

<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/golden_disk_cover_scanlines.jpg?raw=true" alt="golden_disk_cover_scanlines.jpg"/>

Notice that the waveforms are now simplified, where the initial time-series signal is represented with round dots while the bigger sequence is still labelled with the same binary pattern. If you look below that, you will notice the same binary sequence indicating where each bigger sequence belongs as a set of lines written into rectangular block. Another vital clue are the zig-zag lines showing the scan lines, which indicate how the time-series signal is arranged into our rectangular block, while the binary number just above each scanline shows that each line is coming from a single bigger waveform. The rectangular block is actually indicating a presence of image, while the zig-zag scan lines show how data from audio can be used to reconstruct the original image. Another vital clue is just above the last scan line in the rectangle, showing the symbols: |--------- which is a number in binary - where 1 is represented by symbol | and 0 is indicated by symbol -. The number is 10000000 which when converted to decimal is 512, and is an indication that there are 512 scan lines in a single image.

The last piece in the puzzle is the round circle in the rectangular block:
<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/golden_disk_cover_successimage.jpg?raw=true" alt="golden_disk_cover_successimage.jpg" width="300"/>

This image is intended for an alien who might be trying to reverse engineer our messages and is a vital clue to indicate if they have decoded the message correctly. This cicle is a calibration image, which is the first image on the disk. I think a circle is an excellent choice for calibration as, from signal processing perspective, any phase issues when reading this would result in elliptical shapes that still relatively maintain a cicle's characteristics. It is also quite easy to calibrate for phase correctly by looking at distortions with respect to a true circle, which is also a universal shape of most planets and stars in our universe.

# Reading data and looking for clues
Looking at the data digitized and process by the authors of: [^rf1], we can start analysing and applying our clues.

The data can be read and as noted in the repository linked about we need to flip the signal amplitude to get a signal close to the one in the instructions:

```python
# load data and invert signals
in_wav_path = 'data/voyager_left_channel_32bitfloat_384kHz.wav'
rate, data = scipy.io.wavfile.read(in_wav_path)

# correct for waveforms
data = -data
```

Visualizing part of the signal and looking around time step 5999900, we can see how the signal looks like and start comparing it with the ones provided as instructions to the aliens:

<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/initial_signal.png?raw=true" alt="golden_disk_cover_waveform.jpg" width="400"/>

Comparing this with the first instructions given in the figure:

<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/golden_disk_cover_waveform.jpg?raw=true" alt="golden_disk_cover_waveform.jpg" width="400"/>

 we can notice the data between a pair of impulse signals correspond to a single scan line in any given image:
 
<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/initial_signal_edited.png?raw=true" alt="initial_signal_edited.png" width="400"/>

Now to reconstruct the image, all we have to do it extract each scanline and place it into a matrix to get an image.

## Proposing the use of signal peak detector
Looking at previous work at [^rf1] the major issue in reconstructing the images was alignment of multiple signals in an image, requring the use of a simple offset based reading algorithm. The issues faced by the authors were due to non-identical reading speeds at different parts of the disk. 

Observing the signals above, we notice the main part before a signal starts and ends is an impulse peak signal. Using this observation, we can come up with a much more robust algorithm that is based on detecting these peaks and then identifying the signals in between as a single scan-line.

We can look at the following section of the signal to demonstrate how a peak detection algorithm might work. 

```python
data = signal[i: i + window_size]
plot(data)
```

<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/small_signal.png?raw=true" alt="small_signal.png" width="400"/>
<br><br>


To detect the peaks, the first thing we do is to take a derivative of the signal. As the signal is discrete, we can achieve this just by a shift and subtraction from original signal to get gradient at each time-point. We use absolute values so as we are only interested in where peaks exist and not whether they are positive or negative:

```python
data_deriv = np.zeros_like(data)
data_deriv[0:-2] = data[1:-1]
data_deriv = np.abs(data-data_deriv)
plot(data_deriv)
```

<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/small_signal_deriv_abs.png?raw=true" alt="small_signal_deriv_abs.png" width="400"/>
<br><br>

As you can see, this results in multiple peaks in the area where we were expecting a single one. We can process the signal to remove any high frequencies by passing it through a low-pass filter:

```python
data_deriv = filter_signal(data_deriv, filter_size=21)
plot(data_deriv)
data_deriv = (data_deriv>0.008).astype(np.int)
check, indices = check_for_two_peaks(data_deriv)
```

<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/small_signal_deriv_filt.png?raw=true" alt="small_signal_deriv_filt.png" width="400"/>
<br><br>

Finally, we can apply a simple threshold to get locations of the two peaks and use them for reading the relevant data at later steps:

```python
data_deriv = (data_deriv>0.008).astype(np.int)
plot(data_deriv)
check, indices = check_for_two_peaks(data_deriv)
```

<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/small_signal_deriv_step.png?raw=true" alt="small_signal_deriv_step.png" width="400"/>

The final step is to write scanlines into an image plane. THis helps visualize image data as time-series, where at each time-step a new scan line is added. This is visualized as gif below:

![image](https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/imageGif.gif)

## Simple linear resize function
Using the above approach produces scan lines that are of different sizes and require resizing. For this purpose, I implemented a simple linear sampling method. The algorithm defines a new index based on the input and required output shapes. These indices are then used to get weighted sum of the two closest signals from input. An anti-aliasing (low-pass) filter is also introduced to remove any aliasing artefacts:

```python
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
```
## Additional helper functions
Here is a collection of additional helper functions used in this decoding script.

### Low-pass filtering
```python
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
```
### TV-like playback of time-series scanlines
```python
# run things like TV playback - adds single line at a time
def add_image_line(image_data, line):
    scan_width, lines = image_data.shape
    
    # go and shift the scan lines to make space for new line
    for i in range(511):
        image_data[:, i] = image_data[:, i+1]

    # add new line
    image_data[:, -1] = line
    
    return image_data
```

### Two peaks detector
```python
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
```
### Renormalize pixel values for writing to image
```python
# renormalize to pixel values
def renormalize(line):
    min_val = -0.1#np.min(line)
    max_val = 0.1#np.max(line)
    line = np.clip(line, min_val, max_val)
    
    return ((line - min_val)/(max_val-min_val))*255
```
### Plot signal
```python
# plot signal helper
def plot(line):
    plt.plot(range(line.size), line)
    plt.ylim([-0.4, 0.4])
    plt.show()
```

Disclaimer: Where applicable, the images and audio clips are copyright from their respective owners. This tool demonstrates the functionality of reading the image stream, without sharing the actual data/copyrighted material. The audio data was only used for personal research purpose.

[^rf1]:[GitHub - aizquier/voyagerimb: Voyager's Golden Disk Image Browser](https://github.com/aizquier/voyagerimb)
