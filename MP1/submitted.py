'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import numpy as np
import math

def make_frames(signal, hop_length, win_length):
    '''
    frames = make_frames(signal, hop_length, win_length)

    signal (num_samps) - the speech signal
    hop_length (scalar) - the hop length, in samples
    win_length (scalar) - the window length, in samples
    frames (num_frames, win_length) - array with one frame per row

    num_frames should be enough so that each sample from the signal occurs in at least one frame.
    The last frame may be zero-padded.
    '''

    no_of_frames = math.floor(((len(signal)-hop_length)/(win_length-hop_length)))
    frames = np.zeros((no_of_frames, win_length))

    start = 0
    for i in range(no_of_frames):
        frames[i] = (signal[start:start+win_length])
        start += hop_length

    return frames

def correlate(frames):
    '''
    autocor = correlate(frames)

    frames (num_frames, win_length) - array with one frame per row
    autocor (num_frames, 2*win_length-1) - each row is the autocorrelation of one frame
    '''
    autocor = np.zeros((len(frames), (2*len(frames[0])) - 1))
    for i in range(len(frames)):
        autocor[i] = np.correlate(frames[i], frames[i], mode='full')

    return autocor

def make_matrices(autocor, p):
    '''
    R, gamma = make_matrices(autocor, p)

    autocor (num_frames, 2*win_length-1) - each row is symmetric autocorrelation of one frame
    p (scalar) - the desired size of the autocorrelation matrices
    R (num_frames, p, p) - p-by-p Toeplitz autocor matrix of each frame, with R[0] on main diagonal
    gamma (num_frames, p) - length-p autocor vector of each frame, R[1] through R[p]
    '''
    R = np.zeros((len(autocor), p, p))
    gamma = np.zeros((len(autocor), p))
    win_length = int((np.shape(autocor)[1] + 1)/2)

    for i in range(len(autocor)):
        gamma[i] = autocor[i, win_length:win_length+p]
        for j in range(p):
            R[i, j] = autocor[i, win_length-1-j:win_length-1-j+p]

    return R, gamma

def lpc(R, gamma):
    '''
    a = lpc(R, gamma)
    Calculate the LPC coefficients in each frame

    R (num_frames, p, p) - p-by-p Toeplitz autocor matrix of each frame, with R[0] on main diagonal
    gamma (num_frames, p) - length-p autocor vector of each frame, R[1] through R[p]
    a (num_frames,p) - LPC predictor coefficients in each frame
    '''
    a = np.zeros((R.shape[0], R.shape[1]))
    for i in range(len(a)):
        R_inv = np.linalg.inv(R[i])
        a[i] = np.dot(R_inv, gamma[i])

    return a

def framepitch(autocor, Fs):
    '''
    framepitch = framepitch(autocor, samplerate)

    autocor (num_frames, 2*win_length-1) - autocorrelation of each frame
    Fs (scalar) - sampling frequency
    framepitch (num_frames) - estimated pitch period, in samples, for each frame, or 0 if unvoiced

    framepitch[t] = 0 if the t'th frame is unvoiced
    framepitch[t] = pitch period, in samples, if the t'th frame is voiced.
    Pitch period should maximize R[framepitch]/R[0], in the range 4ms <= framepitch < 13ms.
    Call the frame voiced if and only if R[framepitch]/R[0] >= 0.3, else unvoiced.
    '''
    autocor_shape = np.shape(autocor)
    win_length = int(autocor_shape[1]/2)
    framepitch = np.zeros(autocor_shape[0])
    lower_bound = int(math.ceil(Fs*0.004)) + win_length
    upper_bound = int(math.floor(Fs*0.013)) + win_length

    for i in range(autocor_shape[0]):
        C = np.zeros(autocor_shape[0])
        C = autocor[i, lower_bound:upper_bound+1]/autocor[i, win_length]

        max_index = np.argmax(C)
        if C[max_index] < 0.3:
            max_index = 0
        else:
            max_index += int(math.ceil(Fs*0.004))

        framepitch[i] = max_index

    print(framepitch.max())
    return framepitch

def framelevel(frames):
    '''
    framelevel = framelevel(frames)

    frames (num_frames, win_length) - array with one frame per row
    framelevel (num_frames) - framelevel[t] = power (energy/duration) of the t'th frame, in decibels
    '''
    frames_shape = np.shape(frames)
    framelevel = np.zeros(frames_shape[0])

    for i in range(frames_shape[0]):
        framelevel[i] = 10 * np.log10(sum(np.square(frames[i]))/frames_shape[1])

    return framelevel

def interpolate(framelevel, framepitch, hop_length):
    '''
    samplelevel, samplepitch = interpolate(framelevel, framepitch, hop_length)

    framelevel (num_frames) - levels[t] = power (energy/duration) of the t'th frame, in decibels
    framepitch (num_frames) - estimated pitch period, in samples, for each frame, or 0 if unvoiced
    hop_length  (scalar) - number of samples between start of each frame
    samplelevel ((num_frames-1)*hop_length+1) - linear interpolation of framelevel
    samplepitch ((num_frames-1)*hop_length+1) - modified linear interpolation of framepitch

    samplelevel is exactly as given by numpy.interp.
    samplepitch is modified so that samplepitch[n]=0 if the current frame or next frame are unvoiced.
    '''
    samplelevel = np.zeros(((len(framelevel-1) * hop_length) + 1))
    samplepitch = np.zeros(((len(framelevel-1) * hop_length) + 1))
    x_len = np.arange(0, len(framelevel))
    print(x_len)
    print(hop_length)



    #for i in range(len(framepitch)):
        #if framepitch[i] == 0 or framepitch[i+1] == 0:
            #samplepitch[i] = 0

    return samplelevel, samplepitch

def excitation(samplelevel, samplepitch):
    '''
    phase, excitation = excitation(samplelevel, samplepitch)

    samplelevel ((num_frames-1)*hop_length+1) - effective level (in dB) of every output sample
    samplepitch ((num_frames-1)*hop_length+1) - effective pitch period for every output sample
    phase ((num_frames-1)*hop_length+1) - phase of the fundamental at every output sample,
      modulo 2pi, so that 0 <= phase[n] < 2*np.pi for every n.
    excitation ((num_frames-1)*hop_length+1) - LPC excitation signal
      if samplepitch[n]==0, then excitation[n] is zero-mean Gaussian
      if samplepitch[n]!=0, then excitation[n] is a delta function time-aligned to the phase
      In either case, excitation is scaled so its average power matches samplelevel[n].
    '''
    ######## WARNING: the following lines must remain, so that your random numbers will match the grader
    from numpy.random import Generator, PCG64
    rg = Generator(PCG64(1234))
    ## Your Gaussian random numbers must be generated using the command ***rg.normal***
    ## (See https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html).
    ## (1) You must generate them in order, from the beginning to the end of the waveform.
    ## (2) You must generate a random sample _only_ if the corresponding samplepitch[n] > 0.
    raise RuntimeError("You need to write this part!")

def synthesize(excitation, a):
    '''
    y = synthesize(excitation, a)
    excitation ((num_frames-1)*hop_length+1) - LPC excitation signal
    a (num_frames,p) - LPC predictor coefficients in each frame
    y ((num_frames-1)*hop_length+1) - LPC synthesized  speech signal
    '''
    raise RuntimeError("You need to write this part!")
