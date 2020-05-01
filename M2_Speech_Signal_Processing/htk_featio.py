"""Functions to manages features in HTK format."""
import struct
import sys

import numpy as np


def write_htk_user_feat(x, name='filename'):
    """Store features in HTK format."""
    default_period = 100000  # assumes 0.010 ms frame shift
    num_dim = x.shape[0]
    num_frames = x.shape[1]

    # Create header struct
    hdr = struct.pack(
        '>IIHH',  # the beginning '>' says write big-endian
        num_frames,  # nSamples
        default_period,  # samplePeriod
        4 * num_dim,  # 2 floats per feature
        9)  # user features

    out_file = open(name, 'wb')
    out_file.write(hdr)

    for t in range(num_frames):
        frame = np.array(x[:, t], 'f')
        if sys.byteorder == 'little':
            frame.byteswap(True)
        frame.tofile(out_file)

    out_file.close()


def read_htk_user_feat(name='filename'):
    """Retrieve features stored in HTK format."""
    f = open(name, 'rb')
    hdr = f.read(12)
    num_frames, _, samp_size, parm_kind = struct.unpack(">IIHH", hdr)
    if parm_kind != 9:
        raise RuntimeError(
            'feature reading code only validated for USER '
            'feature type for this lab. There is other publicly available '
            'code for general purpose HTK feature file I/O\n')

    num_dim = samp_size // 4

    feat = np.zeros([num_dim, num_frames], 'f')
    for t in range(num_frames):
        feat[:, t] = np.array(
            struct.unpack('>' + ('f' * num_dim), f.read(samp_size)),
            dtype=float)

    return feat


def write_ascii_stats(x, name='filename'):
    """Write ASCII statistics."""
    out_file = open(name, 'w')
    for t in range(0, x.shape[0]):
        out_file.write("{0}\n".format(x[t]))
    out_file.close()
