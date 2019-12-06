from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.utils import db_to_float
import shutil
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import wave

def view(fn, thresh, silence_len):
    spf = wave.open(fn,'r')

    # Extract Raw Audio from Wav File
    # signal = spf.readframes(-1)
    # signal = np.fromstring(signal, 'Int16')

    audio = AudioSegment.from_wav(fn)
    audio_volume = []
    for i in range(len(audio)):
        audio_volume.append(audio[i:i+silence_len].rms)

    # plt.figure(1)
    print(max(audio_volume))
    plt.axhline(db_to_float(thresh)*audio.max_possible_amplitude)
    plt.plot(audio_volume)
    plt.show()

def splitter(main_dir, dir, percent_testing, silence_thresh=-27, silence_len=1, save="False"):
    dir_testing = "{}/{}_testing".format(main_dir,dir)
    dir_training = "{}/{}_training".format(main_dir,dir)
    fn = "{}.wav".format(dir)

    tapping = AudioSegment.from_wav(fn)

    chunks = split_on_silence(tapping,
        # must be silent for at least x ms
        min_silence_len=silence_len,

        # consider it silent if quieter than x dBFS
        # adjust this per requirement
        silence_thresh=silence_thresh,
        keep_silence=10,
        seek_step=1
    )

    view(fn, silence_thresh, silence_len)

    if save == "False" or not save:
        print("No files generated")
        return

    try:
        shutil.rmtree(dir_testing)
        shutil.rmtree(dir_training)
    except:
        pass

    try:
        os.makedirs(dir_testing)
        os.makedirs(dir_training)
    except:
        pass

    # process each chunk per requirements
    for i, chunk in enumerate(chunks):
        if float(i)/len(chunks) < percent_testing:
            export_dir = dir_training
        else:
            export_dir = dir_testing
        # export audio chunk with new bitrate
        print("exporting {}/{}_{}.wav".format(export_dir, dir, i))
        chunk.export("{}/{}_{}.wav".format(export_dir, dir, i), bitrate='192k', format="wav")

        if i <= 5:
            view("{}/{}_{}.wav".format(export_dir, dir, i), silence_thresh, silence_len)

if __name__ == '__main__':

    main_dir = str(sys.argv[1])
    dir = str(sys.argv[2])
    percent_testing = float(sys.argv[3])
    silence_thresh = int(sys.argv[4])
    silence_len = int(sys.argv[5])
    save = sys.argv[6]

    # sample: py silence_trimmer.py tapping_data metal_tapping .5 -27 3 True
    # py silence_trimmer.py tapping_data tape_tapping .5 -27 3 False

    splitter(main_dir, dir, percent_testing, silence_thresh, silence_len, save)
