"""Calculate features for a single file."""
import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import htk_featio as htk
import speech_sigproc as sp

data_dir = '../Experiments'
wav_file = '../LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac'
feat_file = os.path.join(data_dir, 'feat/1272-128104-0000.feat')
plot_output = True

if not os.path.isfile(wav_file):
    raise RuntimeError('input wav file is missing. Have you downloaded the '
                       'LibriSpeech corpus?')

if not os.path.exists(os.path.join(data_dir, 'feat')):
    os.mkdir(os.path.join(data_dir, 'feat'))

samp_rate = 16000

x, s = sf.read(wav_file)
if s != samp_rate:
    raise RuntimeError("LibriSpeech files are 16000 Hz, found {0}".format(s))

fe = sp.FrontEnd(samp_rate=samp_rate, mean_norm_feat=True)

feat = fe.process_utterance(x)

if plot_output:
    if not os.path.exists('fig'):
        os.mkdir('fig')

    # plot waveform
    plt.plot(x)
    plt.title('waveform')
    plt.savefig('fig/waveform.png', bbox_inches='tight')
    plt.close()

    # plot waveform after pre-emphasis
    xp = fe.pre_emphasize(x)
    plt.plot(xp)
    plt.title('waveform with pre-emphasis')
    plt.savefig('fig/waveform-pre.png', bbox_inches='tight')
    plt.close()

    # plot the mag spectrum
    frames = fe.wav_to_frames(x)
    magspec = fe.frames_to_magspec(frames)
    # plt.imshow(np.log(magspec + 1e-7), cmap='viridis', origin='lower')
    plt.imshow(np.log(magspec + 1e-7), origin='lower')
    plt.title('Log Magnitude Spectrogram')
    plt.savefig('fig/log_mag_spec.png', bbox_inches='tight')
    plt.close()

    # plot the mag spectrum after pre-emphasis
    frames = fe.wav_to_frames(xp)
    magspec = fe.frames_to_magspec(frames)
    # plt.imshow(np.log(magspec + 1e-7), cmap='viridis', origin='lower')
    plt.imshow(np.log(magspec + 1e-7), origin='lower')
    plt.title('Log Magnitude Spectrogram After Pre-emphasis')
    plt.savefig('fig/log_mag_spec_p.png', bbox_inches='tight')
    plt.close()

    # plot mel filterbank
    for i in range(0, fe.num_mel):
        plt.plot(fe.mel_filterbank[i, :])
    plt.title('mel filterbank')
    plt.savefig('fig/mel_filterbank.png', bbox_inches='tight')
    plt.close()

    # plot log mel spectrum (fbank)
    # flip the image so that vertical frequency axis goes from low to high
    plt.imshow(feat, origin='lower', aspect=4)
    plt.title('log mel filterbank features (fbank)')
    plt.savefig('fig/fbank.png', bbox_inches='tight')
    plt.close()

htk.write_htk_user_feat(feat, feat_file)
print(f"Wrote {feat.shape[1]} frames to {feat_file}")

# Verify that the file was written correctly
feat2 = htk.read_htk_user_feat(name=feat_file)
print(f"Read {feat2.shape[1]} frames from {feat_file}")
print(
    f"Features are {'the same' if np.allclose(feat, feat2) else 'different'}")
print(
    f"Average Frobenius norm difference:  "
    f"{np.linalg.norm(feat - feat2) / (feat.shape[0] * feat.shape[1]):.2e}")
