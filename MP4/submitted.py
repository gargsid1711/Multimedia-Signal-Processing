import os, h5py
from scipy.stats import multivariate_normal
import numpy as  np

###############################################################################
# A possibly-useful utility function
def compute_features(waveforms, nceps=25):
    '''Compute two types of feature matrices, for every input waveform.

    Inputs:
    waveforms (dict of lists of (nsamps) arrays):
        waveforms[y][n] is the n'th waveform of class y
    nceps (scalar):
        Number of cepstra to retain, after windowing.

    Returns:
    cepstra (dict of lists of (nframes,nceps) arrays):
        cepstra[y][n][t,:] = windowed cepstrum of the t'th frame of the n'th waveform of the y'th class.
    spectra (dict of lists of (nframes,nceps) arrays):
        spectra[y][n][t,:] = liftered spectrum of the t'th frame of the n'th waveform of the y'th class.

    Implementation Cautions:
        Computed with 200-sample frames with an 80-sample step.  This is reasonable if sample_rate=8000.
    '''
    cepstra = { y:[] for y in waveforms.keys() }
    spectra = { y:[] for y in waveforms.keys() }
    for y in waveforms.keys():
        for x in waveforms[y]:
            nframes = 1+int((len(x)-200)/80)
            frames = np.stack([ x[t*80:t*80+200] for t in range(nframes) ])
            spectrogram = np.log(np.maximum(0.1,np.absolute(np.fft.fft(frames)[:,1:100])))
            cepstra[y].append(np.fft.fft(spectrogram)[:,0:nceps])
            spectra[y].append(np.real(np.fft.ifft(cepstra[y][-1])))
            cepstra[y][-1] = np.real(cepstra[y][-1])
    return cepstra, spectra

###############################################################################
# TODO: here are the functions that you need to write
def initialize_hmm(X_list, nstates):
    '''Initialize hidden Markov models by uniformly segmenting input waveforms.

    Inputs:
    X_list (list of (nframes[n],nceps) arrays):
        X_list[n][t,:] = feature vector, t'th frame of n'th waveform, for 0 <= t < nframes[n]
    nstates (scalar):
        the number of states to initialize

    Returns:
    A (nstates,nstates):
        A[i,j] = p(state[t]=j | state[t-1]=i), estimates as
        (# times q[t]=j and q[t-1]=i)/(# times q[t-1]=i).
    Mu (nstates,nceps):
        Mu[i,:] = mean vector of the i'th state, estimated as
        average of the frames for which q[t]=i.
    Sigma (nstates,nceps,nceps):
        Sigma[i,:,:] = covariance matrix, i'th state, estimated as
        unbiased sample covariance of the frames for which q[t]=i.

    Function:
    Initialize the initial HMM by dividing each feature matrix uniformly into portions for each state:
    state i gets X_list[n][int(i*nframes[n]/nstates):int((i+1)*nframes[n]/nstates,:] for all n.
    Then, from this initial uniform alignment, estimate A, MU, and SIGMA.

    Implementation Cautions:
    - For the last state, (# times q[t-1]=i) is not the same as (# times q[t]=i).
    - "Unbiased" means that you divide by N-1, not N.  In np.cov, that means "bias=False".
    '''
    # print(np.shape(X_list)) #(200, )
    nwaves = len(X_list) # number of waveforms = 200
    # print(X_list[0][0: 5, :])
    # print(X_list[0].shape, X_list[0][0].shape) #(nframes[n], 25)
    nceps = len(X_list[0][0]) # (scalar) length of cepstra per waveform in X_list = 25


    nframes = np.zeros(len(X_list)) #list of lengths of waveforms in X_list (each waveform has different number of frames)
    for n in range(len(nframes)):
        nframes[n] = np.shape(X_list[n])[0]

    # print(len(X_list[:][int(0*nframes[0]/nstates):int((0+1)*nframes[0]/nstates),:])) #divides X_list[n] into 5 equal parts (0 when i = 5), length of segments different for each waveform
    length = 0
    for n in range(len(nframes)):
        length += np.shape(X_list[n][int(0*nframes[n]/nstates):int((0+1)*nframes[n]/nstates),:])[0]
    # print(length)
    #
    # states = np.zeros((nstates, nwaves))
    # for i in range(nstates):
    #     for j in range(nwaves):
    #         state_i =[]
    #         for n in range(len(nframes)):
    #         # length = len(X_list[n][int(i*nframes[n]/nstates):int((i+1)*nframes[n]/nstates),:])
    #             # print(X_list[n][int(i*nframes[n]/nstates):int((i+1)*nframes[n]/nstates),:])
    #             # print()
    #             np.append(state_i, X_list[n][int(i*nframes[n]/nstates):int((i+1)*nframes[n]/nstates),:])
            # states[i, j, np.newaxis] = state_i

    # print(states)
    #
    #
    # A = np.zeros((nstates, nstates))
    # for i in range(nstates):
    #     for j in range(nstates):
    #         A[i, j] =

    return

def observation_pdf(X, Mu, Sigma):
    '''Calculate the log observation PDFs for every frame, for every state.

    Inputs:
    X (nframes,nceps):
        X[t,:] = feature vector, t'th frame of n'th waveform, for 0 <= t < nframes[n]
    Mu (nstates,nceps):
        Mu[i,:] = mean vector of the i'th state
    Sigma (nstates,nceps,nceps):
        Sigma[i,:,:] = covariance matrix, i'th state

    Returns:
    B (nframes,nstates):
        B[t,i] = max(p(X[t,:] | Mu[i,:], Sigma[i,:,:]), 1e-100)

    Function:
    The observation pdf, here, should be a multivariate Gaussian.
    You can use scipy.stats.multivariate_normal.pdf.
    '''
    nframes = len(X)
    nstates = len(Mu)

    B = np.zeros((nframes, nstates))
    for t in range(nframes):
        for i in range(nstates):
            B[t, i] = max(multivariate_normal.pdf(X[t, :], Mu[i, :], Sigma[i, :, :]), 10**-100)

    return B

def scaled_forward(A, B):
    '''Perform the scaled forward algorithm.

    Inputs:
    A (nstates,nstates):
        A[i,j] = p(state[t]=j | state[t-1]=i)
    B (nframes,nstates):
        B[t,i] = p(X[t,:] | Mu[i,:], Sigma[i,:,:])

    Returns:
    Alpha_Hat (nframes,nstates):
        Alpha_Hat[t,i] = p(q[t]=i | X[:t,:], A, Mu, Sigma)
        (that's its definition.  That is not the way you should compute it).
    G (nframes):
        G[t] = p(X[t,:] | X[:t,:], A, Mu, Sigma)
        (that's its definition.  That is not the way you should compute it).

    Function:
    Assume that the HMM starts in state q_1=0, with probability 1.
    With that assumption, implement the scaled forward algorithm.
    '''

    nframes = len(B)
    nstates = len(A)

    Alpha = np.zeros((nframes, nstates))
    G = np.zeros(nframes)
    Alpha_Hat = np.zeros((nframes, nstates))

    Alpha[0, 0] = B[0, 0]
    G[0] = np.sum(Alpha[0, :])
    Alpha_Hat[0, :] = Alpha[0, :]/G[0]

    for t in range(1, nframes):
        for j in range(nstates):
            Alpha[t, j] = np.sum(Alpha_Hat[t - 1, :] * A[:, j] * B[t, j])
        G[t] = np.sum(Alpha[t, :])
        for j2 in range(nstates):
            Alpha_Hat[t, j2] = Alpha[t, j2]/G[t]

    return Alpha_Hat, G

def scaled_backward(A, B):
    '''Perform the scaled backward algorithm.

    Inputs:
    A (nstates,nstates):
        A[y][i,j] = p(state[t]=j | state[t-1]=i)
    B (nframes,nstates):
        B[t,i] = p(X[t,:] | Mu[i,:], Sigma[i,:,:])

    Returns:
    Beta_Hat (nframes,nstates):
        Beta_Hat[t,i] = p(X[t+1:,:]| q[t]=i, A, Mu, Sigma) / max_j p(X[t+1:,:]| q[t]=j, A, Mu, Sigma)
        (that's its definition.  That is not the way you should compute it).
    '''

    nframes = len(B)
    nstates = len(A)
    Beta_Hat = np.zeros((nframes, nstates))
    Beta = np.zeros((nframes, nstates))
    C = np.zeros(nframes)

    Beta_Hat[-1, :] = 1

    for t in range(nframes-2, -1, -1):
        for i in range(nstates-1, -1, -1):
            Beta[t, i] = np.sum(A[i, :] * B[t+1, :] * Beta_Hat[t+1, :])
            # print(i)
        C[t] = np.max(Beta[t,:])
        for i2 in range(nstates-1, -1, -1):
            Beta_Hat[t, i2] = Beta[t, i2]/C[t]

    return Beta_Hat

def posteriors(A, B, Alpha_Hat, Beta_Hat):
    '''Calculate the state and segment posteriors for an HMM.

    Inputs:
    A (nstates,nstates):
        A[y][i,j] = p(state[t]=j | state[t-1]=i)
    B (nframes,nstates):
        B[t,i] = p(X[t,:] | Mu[i,:], Sigma[i,:,:])
    Alpha_Hat (nframes,nstates):
        Alpha_Hat[t,i] = p(q=i | X[:t,:], A, Mu, Sigma)
    Beta_Hat (nframes,nstates):
        Beta_Hat[t,i] = p(X[t+1:,:]| q[t]=i, A, Mu, Sigma) / prod(G[t+1:])

    Returns:
    Gamma (nframes,nstates):
        Gamma[t,i] = p(q[t]=i | X, A, Mu, Sigma)
                   = Alpha_Hat[t,i]*Beta_Hat[t,i] / sum_i numerator
    Xi (nframes-1,nstates,nstates):
        Xi[t,i,j] = p(q[t]=i, q[t+1]=j | X, A, Mu, Sigma)
                  = Alpha_Hat[t,i]*A{i,j]*B[t+1,j]*Beta_Hat[t+1,j] / sum_{i,j} numerator


    Implementation Warning:
    The denominators, in either Gamma or Xi, might sometimes become 0 because of roundoff error.
    YOU MUST CHECK FOR THIS!
    Only perform the division if the denominator is > 0.
    If the denominator is == 0, then don't perform the division.
    '''

    nframes = len(B)
    nstates = len(A)

    Gamma = np.zeros((nframes, nstates))
    Xi = np.zeros((nframes-1, nstates, nstates))

    for t in range(nframes):
        for i in range(nstates):
            Gamma[t, i] = (Alpha_Hat[t, i] * Beta_Hat[t, i])/np.sum(Alpha_Hat[t, :] * Beta_Hat[t, :])

    for t in range(nframes-1):
        for i in range(nstates):
            for j in range(nstates):
                denominator = np.sum(Alpha_Hat[t, :] * np.sum(B[t+1, :] * Beta_Hat[t+1, :] * A, axis=1))
                Xi[t, i, j] = (Alpha_Hat[t, i] * A[i, j] * B[t+1, j] * Beta_Hat[t+1, j])
                if denominator != 0:
                    Xi[t, i, j] /= denominator



    return Gamma, Xi

def E_step(X, Gamma, Xi):
    '''Calculate the expectations for an HMM.

    Inputs:
    X (nframes,nceps):
        X[t,:] = feature vector, t'th frame of n'th waveform
    Gamma (nframes,nstates):
        Gamma[t,i] = p(q[t]=i | X, A, Mu, Sigma)
    Xi (nsegments,nstates,nstates):
        Xi_list[t,i,j] = p(q[t]=i, q[t+1]=j | X, A, Mu, Sigma)
        WARNING: rows of Xi may not be time-synchronized with the rows of Gamma.

    Returns:
    A_num (nstates,nstates):
        A_num[i,j] = E[# times q[t]=i,q[t+1]=j]
    A_den (nstates):
        A_den[i] = E[# times q[t]=i]
    Mu_num (nstates,nceps):
        Mu_num[i,:] = E[X[t,:]|q[t]=i] * E[# times q[t]=i]
    Mu_den (nstates):
        Mu_den[i] = E[# times q[t]=i]
    Sigma_num (nstates,nceps,nceps):
        Sigma_num[i,:,:] = E[(X[t,:]-Mu[i,:])@(X[t,:]-Mu[i,:]).T|q[t]=i] * E[# times q[t]=i]
    Sigma_den (nstates):
        Sigma_den[i] = E[# times q[t]=i]
    '''

    nframes = len(X)
    nceps = X.shape[1]
    nstates = Gamma.shape[1]

    A_num = np.zeros((nstates, nstates))
    A_den = np.zeros(nstates)
    Mu_num = np.zeros((nstates, nceps))
    Mu_den = np.zeros(nstates)
    Mu = np.zeros(nceps)
    Sigma_curr = np.zeros((nceps, nceps))
    Sigma_num = np.zeros((nstates, nceps, nceps))
    Sigma_den = np.zeros(nstates)

    # print((X-Mu).shape) = (nframes, 25)
    # print(np.dot(np.transpose(X - Mu), (X - Mu)).shape) (25x25)
    # print(np.dot((X - Mu), np.transpose(X - Mu)).shape) (nframes,nfrmaes)
    # print(Gamma[:, i].shape) nframesx1

    for i in range(nstates):
        Mu_num[i, :] = np.sum(np.transpose(np.transpose(X) * Gamma[:, i]), axis=0)
        Mu_den[i] = np.sum(Gamma[:, i])
        Mu = Mu_num[i, :]/Mu_den[i] #mu.shape = 1x25
        Sigma_curr = 0
        for t in range(nframes):
            Sigma_curr += np.sum(Gamma[t, i]) * np.outer((X[t, :] - Mu), np.transpose(X[t, :] - Mu))
        # Sigma_num[i, :, :] = np.sum(Gamma[:, i]) * np.outer((X - Mu), np.transpose(X - Mu)) #sigma.shape = 5x25x25, X.shape = nframesx25
        Sigma_num[i, :, :] = Sigma_curr
        Sigma_den[i] = np.sum(Gamma[:, i])

        A_den_curr = 0
        for j in range(nstates):
            A_num[i, j] = np.sum(Xi[:, i, j])
            A_den_curr += np.sum(Xi[:, i, j])

        A_den[i] = A_den_curr


    return A_num, A_den, Mu_num, Mu_den, Sigma_num, Sigma_den

def M_step(A_num, A_den, Mu_num, Mu_den, Sigma_num, Sigma_den, regularizer):
    '''Perform the M-step for an HMM.

    Inputs:
    A_num (nstates,nstates):
        A_num[i,j] = E[# times q[t]=i,q[t+1]=j]
    A_den (nstates):
        A_den[i] = E[# times q[t]=i]
    Mu_num (nstates,nceps):
        Mu_num[i,:] = E[X[t,:]|q[t]=i] * E[# times q[t]=i]
    Mu_den (nstates):
        Mu_den[i] = E[# times q[t]=i]
    Sigma_num (nstates,nceps,nceps):
        Sigma_num[i,:,:] = E[(X[t,:]-Mu[i,:])@(X[t,:]-Mu[i,:]).T|q[t]=i] * E[# times q[t]=i]
    Sigma_den (nstates):
        Sigma_den[i] = E[# times q[t]=i]
    regularizer (scalar):
        Coefficient used for Tikohonov regularization of each covariance matrix.

    Returns:
    A (nstates,nstates):
        A[y][i,j] = p(state[t]=j | state[t-1]=i), estimated as
        E[# times q[t]=j and q[t-1]=i]/E[# times q[t-1]=i)].
    Mu (nstates,nceps):
        Mu[i,:] = mean vector of the i'th state, estimated as
        E[average of the frames for which q[t]=i].
    Sigma (nstates,nceps,nceps):
        Sigma[i,:,:] = covariance matrix, i'th state, estimated as
        E[biased sample covariance of the frames for which q[t]=i] + regularizer*I
    '''

    nstates = len(A_num)
    nceps = Mu_num.shape[1]

    A = np.zeros((nstates, nstates))
    Mu = np.zeros((nstates, nceps))
    Sigma = np.zeros((nstates, nceps, nceps))
    Identity = np.identity(nceps)

    for i in range(nstates):
        A[i] = A_num[i, :]/A_den[i]
        Mu[i] = Mu_num[i, :]/Mu_den[i]
        Sigma[i] = (Sigma_num[i]/Sigma_den[i]) + regularizer * Identity

    return A, Mu, Sigma

def recognize(X, Models):
    '''Perform isolated-word speech recognition using trained Gaussian HMMs.

    Inputs:
    X (list of (nframes[n],nceps) arrays):
        X[n][t,:] = feature vector, t'th frame of n'th waveform
    Models (dict of tuples):
        Models[y] = (A, Mu, Sigma) for class y
        A (nstates,nstates):
             A[i,j] = p(state[t]=j | state[t-1]=i, Y=y).
        Mu (nstates,nceps):
             Mu[i,:] = mean vector of the i'th state for class y
        Sigma (nstates,nceps,nceps):
             Sigma[i,:,:] = covariance matrix, i'th state for class y

    Returns:
    logprob (dict of numpy arrays):
       logprob[y][n] = log p(X[n] | Models[y] )
    Y_hat (list of strings):
       Y_hat[n] = argmax_y p(X[n] | Models[y] )

    Implementation Hint:
    For each y, for each n,
    call observation_pdf, then scaled_forward, then np.log, then np.sum.
    '''
    nframes = len(X)

    logprob = {"1":[],"2":[],"3":[]}

    for y in Models.keys():
        A = Models[y][0]
        Mu = Models[y][1]
        Sigma = Models[y][2]

        for n in range(nframes):
            B = observation_pdf(X[n], Mu, Sigma)
            Alpha_Hat, G = scaled_forward(A, B)
            logprob[y].append(np.sum(np.log(G)))
        logprob[y] = np.array(logprob[y])

    Y_hat = []
    for n in range(nframes):
        for y in Models.keys():
            max = 0
            if logprob[y][n] > max:
                max = logprob[y][n]
                index = y
        Y_hat.append(y)


    return logprob, Y_hat
