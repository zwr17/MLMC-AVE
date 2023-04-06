import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # set gpu number
import numpy as np
import tensorflow as tf
import vggish_input
import vggish_params
import vggish_slim
import h5py
import numpy as np
import resampy
from scipy.io import wavfile

import mel_features
import vggish_params

def waveform_to_examples(data, sample_rate):
  """Converts audio waveform into an array of examples for VGGish.

  Args:
    data: np.array of either one dimension (mono) or two dimensions
      (multi-channel, with the outer dimension representing channels).
      Each sample is generally expected to lie in the range [-1.0, +1.0],
      although this is not required.
    sample_rate: Sample rate of data.

  Returns:
    3-D np.array of shape [num_examples, num_frames, num_bands] which represents
    a sequence of examples, each of which contains a patch of log mel
    spectrogram, covering num_frames frames of audio and num_bands mel frequency
    bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
  """
  # Convert to mono.
  #print(data.shape)#(16000, 8)
  #print(len(data.shape)) #2
  #print(data.ndim)#2d
  if len(data.shape) > 1:
    #data=data.T
    #print(data[-1].shape)#(16000,0)
    #data = data.flatten()
    #print(data.shape)#
    data = np.mean(data, axis=1)
  #print(data.shape)#(16000,)
  # Resample to the rate assumed by VGGish.
  if sample_rate != vggish_params.SAMPLE_RATE:
    data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

  # Compute log mel spectrogram features.
  log_mel = mel_features.log_mel_spectrogram(
      data,
      audio_sample_rate=vggish_params.SAMPLE_RATE,
      log_offset=vggish_params.LOG_OFFSET,
      window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
      hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
      num_mel_bins=vggish_params.NUM_MEL_BINS,
      lower_edge_hertz=vggish_params.MEL_MIN_HZ,
      upper_edge_hertz=vggish_params.MEL_MAX_HZ)

  # Frame features into examples.
  features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
  example_window_length = int(round(
      vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
  example_hop_length = int(round(
      vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
  log_mel_examples = mel_features.frame(
      log_mel,
      window_length=example_window_length,
      hop_length=example_hop_length)
  return log_mel_examples

# Paths to downloaded VGGish files.
checkpoint_path = 'vggish_model.ckpt'
pca_params_path = 'vggish_pca_params.npz'
num_secs = 15 # length of the audio sequence. Videos in our dataset are all 10s long.
freq = 1000
#sr = 44100 #-》八个声道
#16000
sr = 16000


# path of audio files and AVE annotation
audio_dir = "./datasetupdate/audio16bit8c" # .wav audio files
lis = os.listdir(audio_dir)
#print(lis)
#dirs = os.listdir(audio_dir)
#for file in dirs:
#   print(file)
lis.sort(key=lambda x:int(x[:-4]))

#lis = lis.sort(key= lambda x:int(x[:-4])) #按顺序读入
len_data = len(lis)
#print(len_data)
#len_data = 2
audio_features = np.zeros([len_data, 8,15, 128])
audio_features1 = np.zeros([8, 15, 128])

i = 0
for n in range(len_data):

    '''feature learning by VGG-net trained by audioset'''
    #audio_index = os.path.join(audio_dir, lis[n] + '.wav') # path of your audio files
    #print(lis[n])
    audio_index = os.path.join(audio_dir, lis[n] )  # path of your audio files
    print(audio_index)

    sr, wav_data1 = wavfile.read(audio_index)
    # print(audio_index.shape)#(240000, 8)
    # print(audio_index[:,0].shape)#(240000，)
    num_channel = 8
    for c in (0,7):
        wav_data = wav_data1[:,c]

        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        T = 15
        L = wav_data.shape[0]
        log_mel = np.zeros([T, 96, 64])
        # log_mel = np.zeros([T, 8, 96, 64])
        for ii in range(T):
            s = ii * sr
            e = (ii + 1) * sr
            # print(wav_data.shape) #八声道的情况(240000, 8)
            if len(wav_data.shape) > 1:
                data = wav_data[s:e, :]
            else:
                data = wav_data[s:e]
            # print(waveform_to_examples(data, sr).shape)
            log_mel[ii, :, :] = waveform_to_examples(data, sr)
    #return log_mel

        input_batch = log_mel
    #input_batch = vggish_input.wavfile_to_examples(audio_index)
    #print(input_batch.shape)#(15, 96, 64)
    #print(input_batch.shape)
    #print([num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS]) 15 96 64
        np.testing.assert_equal(input_batch.shape,[num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])

    # Define VGGish, load the checkpoint, and run the batch through the model to
    # produce embeddings.
        with tf.Graph().as_default(), tf.Session() as sess:
            vggish_slim.define_vggish_slim()
            vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

            features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)
            [embedding_batch] = sess.run([embedding_tensor],
                                        feed_dict={features_tensor: input_batch})
            #print('VGGish embedding: ', embedding_batch[0])
            #print(embedding_batch.shape)  #(15,128)
            #print(audio_features.shape)
            #audio_features1[i, :, :] = embedding_batch
            #audio_features[i, :, :] = embedding_batch
            #audio_features[i, :, :] （15，512）
            #embedding 15,6,4,51

        audio_features1[c,:,:] = embedding_batch
    #print(audio_features.shape)(num, 8, 15, 128)
    #print(audio_features1.shape)
    audio_features1.transpose((1,0,2)) #(num, 15, 8, 128)
    audio_features[i, :, :,:]= audio_features1
    #print(audio_features.shape)
    i += 1


# save the audio features into one .h5 file. If you have a very large dataset, you may save each feature into one .npy file

with h5py.File('./audio_embedding_8c_update.h5', 'w') as hf:
    hf.create_dataset("dataset",  data=audio_features)

