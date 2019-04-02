# VoyagerImageDecoder
Decoding images from Voyager Disc

![image](https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/imageGif.gif)


# Introduction
The year is 1977, NASA is attempting to send a probe that entails what we as humans are capable of. Most importantly, it captures our ability to dream things that are much bigger than us and will most probably outlast our existence (atleast for some of the visionary scientists who came up with the idea). The two missions are, appropriately, called Voyager I and II as they begin their voyage into the unknown,  attempting to reach the furthest any man-made probe has every been. The big question is raised, if you were to send a preview of our existence as a time capsule to an alien civilization, what would you send? What if you are limited by technology to phonographic audio disks?

Famously know as the golden records due to the gold plating, the voyager records contain audio include greetings in many different languages, along with music and a concise collection of images showing different aspects of human existence. The images had to be encoded in a way so that they could be written as audio clips into one of the two golden disks. 

<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/golden_disk.jpg?raw=true" alt="golden_disk.jpg" width="400"/>


# Instructions for aliens
As bizzare as it sounds, instructions were given to aliens in the form of a disk cover, showing in a very concise and descriptive way - the origins of the probe and what needed to be done to decode the images. I attempt to decode these images by observing the instructions as well as getting help from existing work on voyager golden disk image browser [^rf1]. 
<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/golden_disk_cover.jpg?raw=true" alt="golden_disk_cover.jpg" width="400"/>


## Clues for decoding images
The first clue appears in the top right of the cover where you can see a waveform:
<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/golden_disk_cover_waveform.jpg?raw=true" alt="golden_disk_cover_waveform.jpg" width="400"/>

As noted in [^rf1] the wave form appear to be sequentially written data with each complete waveform representing a sequence, while each waveform being part of a bigger sequence. Notice the binary number below each peaks and valleys in the signal indicating that each one of them correspond to a shorter sequence, while the binary numbers above each waveform indicate that they are part of a bigger sequence. So by now, we have two sequences, 1) smaller and 2) bigger. But what does each sequence represent?

If you look close at the images just below the waveforms, you may be able to find clues on what these are?

<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/golden_disk_cover_scanlines.jpg?raw=true" alt="golden_disk_cover_scanlines.jpg"/>

Notice that the waveforms are now simplified, where the smaller sequence is represented with round dots while the bigger sequence is still labelled with the same binary pattern. If you look below that, you will notice the same binary sequence indicating where each bigger sequence belongs as a set of lines written into rectangular block. Another vital clue are the zig-zag lines showing the scan lines, which indicate how the smaller sequence is arranged into our rectangular block, while the binary number just above each scanline shows that each line is coming from a single bigger waveform. The rectangular block is actually indicating a presence of image, while the zig-zag scan lines show how data from audio can be used to reconstruct the original image.

The last piece in the puzzle is the round circle in the rectangular block:

<img src="https://github.com/devkicks/VoyagerImageDecoder/blob/master/images/golden_disk_cover_successimage.jpg?raw=true" alt="golden_disk_cover_successimage.jpg" width="300"/>

This image is intended for an alien who might be trying to reverse engineering our messages and is a vital clue to indicate if they have decoded the message correctly. This cicle is a calibration image, which is the first image on the disk. I think a circle was an excellent choice here as any phase issues when reading this would result in elliptical shapes that still relatively maintain a cicle's characteristics. It is also quite easy to calibrate for phase correctly by looking at distortions with respect to a true circle, which is also a universal shape of most planets and stars in our universe.

# Reading data and looking for clues
Looking at the data digitized and process by the authors of: [GitHub - aizquier/voyagerimb: Voyager's Golden Disk Image Browser](https://github.com/aizquier/voyagerimb), we can start analysing and applying our clues.

```python
# load data and invert signals
in_wav_path = 'data/voyager_left_channel_32bitfloat_384kHz.wav'
rate, data = scipy.io.wavfile.read(in_wav_path)

# correct for waveforms
data = -data
```

Disclaimer: Where applicable, the images and audio clips are copyright from their respective owners. This tool demonstrates the functionality of reading the image stream, without sharing the actual data/copyrighted material. The audio data was only used for personal research purpose.

[^rf1]:[GitHub - aizquier/voyagerimb: Voyager's Golden Disk Image Browser](https://github.com/aizquier/voyagerimb)

TODO:
Add intro
Limitation of existing software
Add info regarding aligning
Detecting two peaks
Linearly resize
Normalize
check if anything remaining
