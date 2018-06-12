######################################
# This is the code base for automatically obtaining aligned lyrics for solo-singing
# in Smule's Sing! karaoke DAMP dataset. It provides a cleaner annotated dataset.
# Please refer to the following paper for more details:
# Chitralekha Gupta, Rong Tong, Haizhou Li, and Ye Wang, "Automatic Sung-Lyrics Data Annotation"
# Accepted in ISMIR 2018.
######################################

import speech_recognition as sr
import os
import numpy as np
from pydub import AudioSegment
import scipy.signal
import scipy.io.wavfile
import matplotlib.pylab as plt
import re
from levenshtein import *
from httplib import BadStatusLine
import wave
import inflect
from shutil import copyfile

plot = 0
NFFT = 512

def M4AtoWAV16k(M4Afile,WAVfile):
    # Converts .m4a to .wav
    # Input: m4a file, output wavfile name/path
    # Output: wavfile
    #######################
    fs = 16000
    print M4Afile
    m4a_version = AudioSegment.from_file(M4Afile)
    m4a_version.export(WAVfile, format="wav")
    fs_raw,raw_wav_data = scipy.io.wavfile.read(WAVfile)
    n = raw_wav_data.shape[0]
    y = np.floor(np.log2(n))
    nextpow2 = np.power(2, y + 1)
    raw_wav_data = np.pad(raw_wav_data, ((0, nextpow2 - n)), mode='constant')
    resampled_signal = scipy.signal.resample(raw_wav_data/32768.0,fs*len(raw_wav_data)*1.0/fs_raw)
    if max(resampled_signal)>=1.0:
        resampled_signal = resampled_signal*0.9
    scipy.io.wavfile.write(WAVfile,16000,resampled_signal)

    #########################################################################
def InitialFinalSilenceRemoved(sig):
    # Removes beginning and end silence periods of a wavfile
    # Input: sig, i.e. wavfile
    # Output: new_sig, i.e. wavfile without beginning and end silence periods
    #########################################################################
    window = 512
    hop = window/2
    energy = []
    i = 0
    energy_index = []

    while i<(len(sig)-window):
        chunk = sig[i:i+window][np.newaxis]
        energy.append(chunk.dot(chunk.T)[0][0])
        energy_index.append(i)
        i = i+hop

    energy = np.array(energy)
    energy_thresh = 0.1*np.mean(energy)
    significant_indices = np.where(energy>energy_thresh)[0]

    if significant_indices[0] == 0:
        start_point_sample = 0
    else:
        start_point_sample = (significant_indices[0]-1)*hop
    if significant_indices[-1] == len(energy)-1:
        end_point_sample = len(energy)*hop
    else:
        end_point_sample = (significant_indices[-1]+1)*hop
    new_sig = sig[start_point_sample:end_point_sample+1]
    if plot:
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(range(len(sig)),sig)
        plt.ylabel('amplitude')
        plt.title('Remove initial and final silences')
        plt.subplot(3,1,2)
        plt.plot(energy_index,energy)
        plt.ylabel('energy')
        plt.stem([start_point_sample,end_point_sample],[5,5],'k')
        plt.subplot(3,1,3)
        plt.plot(new_sig)
        plt.ylabel('amplitude')
        plt.xlabel('sample number')
        plt.show()
    return new_sig

    #########################################################################
def WavSplit(fs,initialsegmentfolder,file, sig,boundary,hop):
    # Splits wavfile into segments based on boundaries provided
    # Input: fs: sampling frequency
    #        initialsegmentfolder: the folder location where all the wav segments are dumped
    #        file: the wavfile to be split into segments
    #        sig: the wavfile data in an array
    #        boundary: pause-based segment boundaries
    #        hop: the hop size for calculating energy. The boundary values are based on this hop size.
    # Output: wavfile segments from 0 to N in initialsegmentfolder
    #########################################################################

    ind = 0
    start = 0
    for elem in boundary:

        if start==0 and elem!=0:
            segment = sig[start*hop:elem*hop]
            start = elem
        elif start!=0:
            end = elem
            segment = sig[start * hop:end * hop]
            start=elem
        else:
            start = elem
            continue
        ind = ind + 1
        if ind<10:
            WAVfile = initialsegmentfolder+os.sep+file.replace('.wav','_0'+str(ind)+'.wav')
        else:
            WAVfile = initialsegmentfolder + os.sep + file.replace('.wav', '_' + str(ind) + '.wav')
        segment = InitialFinalSilenceRemoved(segment)
        # to avoid less than 2 seconds sounds
        if len(segment)*1.0/fs<2:
            continue

        if max(segment)>=1.0:
            print "exceeding beyond 1.0", max(segment)
            if max(segment)<=1.2:
                segment = segment*0.8
                scipy.io.wavfile.write(WAVfile, fs, np.array(segment * 32678.0, dtype=np.int16))
            else:
                scipy.io.wavfile.write(WAVfile, fs, np.array(segment, dtype=np.int16))
        else:
            scipy.io.wavfile.write(WAVfile, fs, np.array(segment*32678.0,dtype=np.int16))

    #last segment
    segment = sig[elem * hop:]
    ind = ind + 1
    if ind < 10:
        WAVfile = initialsegmentfolder + os.sep + file.replace('.wav', '_0' + str(ind) + '.wav')
    else:
        WAVfile = initialsegmentfolder + os.sep + file.replace('.wav', '_' + str(ind) + '.wav')
    segment = InitialFinalSilenceRemoved(segment)

    if max(segment) >= 1.0:
        print "exceeding beyond 1.0", max(segment)
        if max(segment) <= 1.2:
            segment = segment * 0.8
            scipy.io.wavfile.write(WAVfile, fs, np.array(segment * 32678.0, dtype=np.int16))
        else:
            scipy.io.wavfile.write(WAVfile, fs, np.array(segment, dtype=np.int16))
    else:
        scipy.io.wavfile.write(WAVfile, fs, np.array(segment * 32678.0, dtype=np.int16))

    #########################################################################
def WavSplitMin10sec(fs,initialsegmentfolder,file, sig,boundary,hop):
    # Splits wavfile into 10 seconds segments based on boundaries provided
    # Input: fs: sampling frequency
    #        initialsegmentfolder: the folder location where all the wav segments are dumped
    #        file: the wavfile to be split into segments
    #        sig: the wavfile data in an array
    #        boundary: pause-based segment boundaries
    #        hop: the hop size for calculating energy. The boundary values are based on this hop size.
    # Output: wavfile segments from 0 to N in initialsegmentfolder
    #########################################################################
    ind = 0
    start = 0
    for elem in boundary:

        if start==0 and elem!=0:
            duration = (elem -start)* hop*1.0/fs
            if duration<10:
                continue
            segment = sig[start*hop:elem*hop]
            start = elem
        elif start!=0:
            duration = (elem - start) * hop * 1.0 / fs
            if duration < 10:
                continue
            end = elem
            segment = sig[start * hop:end * hop]
            start=elem
        else:
            start = elem
            continue
        ind = ind + 1
        if ind<10:
            WAVfile = initialsegmentfolder+os.sep+file.replace('.wav','_0'+str(ind)+'.wav')
        else:
            WAVfile = initialsegmentfolder + os.sep + file.replace('.wav', '_' + str(ind) + '.wav')
        segment = InitialFinalSilenceRemoved(segment)
        # to avoid less than 2 seconds sounds
        if len(segment)*1.0/fs<2:
            continue

        if max(segment)>=1.0:
            print "exceeding beyond 1.0", max(segment)
            if max(segment)<=1.2:
                segment = segment*0.8
                scipy.io.wavfile.write(WAVfile, fs, np.array(segment * 32678.0, dtype=np.int16))
            else:
                scipy.io.wavfile.write(WAVfile, fs, np.array(segment, dtype=np.int16))
        else:
            scipy.io.wavfile.write(WAVfile, fs, np.array(segment*32678.0,dtype=np.int16))

    ####last segment###
    duration = (len(sig)-elem*hop)*1.0/fs
    if duration<10: return
    segment = sig[elem * hop:]
    ind = ind + 1
    if ind < 10:
        WAVfile = initialsegmentfolder + os.sep + file.replace('.wav', '_0' + str(ind) + '.wav')
    else:
        WAVfile = initialsegmentfolder + os.sep + file.replace('.wav', '_' + str(ind) + '.wav')
    segment = InitialFinalSilenceRemoved(segment)

    if max(segment) >= 1.0:
        print "exceeding beyond 1.0", max(segment)
        if max(segment) <= 1.2:
            segment = segment * 0.8
            scipy.io.wavfile.write(WAVfile, fs, np.array(segment * 32678.0, dtype=np.int16))
        else:
            scipy.io.wavfile.write(WAVfile, fs, np.array(segment, dtype=np.int16))
    else:
        scipy.io.wavfile.write(WAVfile, fs, np.array(segment * 32678.0, dtype=np.int16))

    #########################################################################
def SplitWavdataByEnergy(sig,fs,initialsegmentfolder,file):
    # Splits wav data based on energy, i.e. pauses
    # Input: sig: the wavfile data in an array
    #        fs: sampling frequency
    #        initialsegmentfolder: the folder location where all the wav segments are dumped
    #        file: the wavfile to be split into segments
    # Output: wavfile segments from 0 to N in initialsegmentfolder
    #########################################################################

    window = 512
    hop = window / 2
    energy = []
    i = 0
    energy_index = []
    while i < (len(sig) - window):
        chunk = sig[i:i + window][np.newaxis]
        energy.append(chunk.dot(chunk.T)[0][0])
        energy_index.append(i)
        i = i + hop

    energy = np.array(energy)
    energy_thresh = 0.1*np.mean(energy) #mean because there might be some spurious peak in energy
    indiceswithlowenergy = np.where(energy <= energy_thresh)
    timeinstance_withlowenergy = indiceswithlowenergy[0] * hop*1.0/fs

    ### retain those silences which are greater than or equal to 0.2 seconds, and hence find valid silent segments
    sil_dur_cap = 0.2
    num_samp_sil_dur_cap = np.floor(sil_dur_cap*fs*1.0/hop)
    lowenergyindices = indiceswithlowenergy[0]

    validlowenergy_subarray = []
    validlowenergyarray = []
    # print lowenergyindices
    for ind in range(len(lowenergyindices)-1):
        diff = lowenergyindices[ind+1]-lowenergyindices[ind]
        if diff>1:
            ##to account for breathy regions## BUT THIS PIECE OF CODE SPLITS FROM CONSONANTS ## NOT desirable
            # if diff>np.floor(0.2*fs*1.0/hop) and diff<np.floor(0.3*fs*1.0/hop): #0.2-0.3 seconds of breathy voice allowed
            #     for i in range(lowenergyindices[ind],lowenergyindices[ind+1],1):
            #         validlowenergy_subarray.append(i)
            #     continue
            #################################
            if validlowenergy_subarray:
                validlowenergy_subarray.append(lowenergyindices[ind])
                if len(validlowenergy_subarray)>=num_samp_sil_dur_cap:
                    validlowenergyarray=validlowenergyarray+validlowenergy_subarray
            validlowenergy_subarray = []
            continue
        validlowenergy_subarray.append(lowenergyindices[ind])
    if len(validlowenergy_subarray) >= num_samp_sil_dur_cap:
        validlowenergyarray = validlowenergyarray + validlowenergy_subarray
    validlowenergy_subarray = []

    #########################
    ##Finding center of valid silent regions. These will be boundaries of phrases/segments/song lines
    boundary = []
    for ind in range(len(validlowenergyarray) - 1):
        diff = validlowenergyarray[ind + 1] - validlowenergyarray[ind]
        if diff > 1:
            if validlowenergy_subarray:
                validlowenergy_subarray.append(validlowenergyarray[ind])
                boundary.append(validlowenergy_subarray[0]+((validlowenergy_subarray[-1]-validlowenergy_subarray[0])/2))
            validlowenergy_subarray = []
            continue
        validlowenergy_subarray.append(validlowenergyarray[ind])
    if validlowenergy_subarray:
        boundary.append(validlowenergy_subarray[0] + ((validlowenergy_subarray[-1] - validlowenergy_subarray[0]) / 2))

    WavSplitMin10sec(fs,initialsegmentfolder,file, sig,boundary,hop)

    ##########################
    if plot:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(range(len(sig)),sig)
        plt.ylabel('amplitude')
        plt.subplot(2, 1, 2)
        plt.plot(energy_index,energy)
        plt.stem(indiceswithlowenergy[0]*hop, 5*np.ones(len(indiceswithlowenergy[0])), 'k')
        plt.ylabel('energy')
        plt.show()

        plt.figure()
        plt.title('amplitude vs. time')
        plt.subplot(2, 1, 1)
        plt.plot(np.array(range(len(sig)))*1.0/fs, sig)
        plt.ylabel('amplitude')
        plt.subplot(2, 1, 2)
        plt.plot(np.array(energy_index)*1.0/fs, energy)
        plt.stem(timeinstance_withlowenergy, 5 * np.ones(len(indiceswithlowenergy[0])), 'k')
        plt.ylabel('energy')
        plt.show()

    if plot:
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(range(len(sig)),sig)
        plt.ylabel('amplitude')
        plt.subplot(3, 1, 2)
        plt.plot(energy_index,energy)
        plt.stem(indiceswithlowenergy[0]*hop, 5*np.ones(len(indiceswithlowenergy[0])), 'k')
        plt.ylabel('energy')
        plt.subplot(3, 1, 3)
        plt.plot(energy_index, energy)
        plt.stem(np.array(validlowenergyarray) * hop, 10 * np.ones(len(validlowenergyarray)), 'k')
        plt.ylabel('energy')
        plt.show()

        plt.figure()
        plt.title('amplitude vs. time')
        plt.subplot(2, 1, 1)
        plt.plot(np.array(range(len(sig)))*1.0/fs, sig)
        plt.ylabel('amplitude')
        plt.subplot(2, 1, 2)
        plt.plot(np.array(energy_index)*1.0/fs, energy)
        plt.stem(np.array(validlowenergyarray) * hop*1.0/fs, 5 * np.ones(len(validlowenergyarray)), 'k')
        plt.ylabel('energy')
        plt.show()

    if plot:
        plt.figure()
        plt.title('amplitude vs. time')
        plt.subplot(2, 1, 1)
        plt.plot(np.array(range(len(sig)))*1.0/fs, sig)
        plt.ylabel('amplitude')
        plt.subplot(2, 1, 2)
        plt.plot(np.array(energy_index)*1.0/fs, energy)
        plt.stem(np.array(validlowenergyarray) * hop*1.0/fs, 1 * np.ones(len(validlowenergyarray)), 'k')
        plt.stem(np.array(boundary) * hop * 1.0 / fs, 10 * np.ones(len(boundary)), 'r')
        plt.ylabel('energy')
        plt.show()
    return

    #########################################################################
def WavSplitter(wavfile,initialsegmentfolder,file):
    # Wrapper for splitting wavfiles
    # Input: wavfile: path to the wavfile to be splitted
    #        initialsegmentfolder: where all the segments will be dumped
    #        file: name of the wavfile
    # Output: wavfile segments to be dumped in initialsegmentfolder
    ################################################################
    fs, wav_data = scipy.io.wavfile.read(wavfile)
    # wav_data = wav_data/32768.0 #NOT doing this step as the wav file written is already between -1 to +1
    wav_data = wav_data - np.mean(wav_data)  # remove DC offset
    wav_data = InitialFinalSilenceRemoved(wav_data)
    SplitWavdataByEnergy(wav_data,fs,initialsegmentfolder,file)

def GetLyrics(lyricsfolder,songname):
    # Get lyrics of a particular song
    # Input: lyricsfolder: where lyrics files of all the songs are kept
    #        songname: the name of the song for which lyrics is needed
    # Output: all_lines: all the lines of the lyrics file in one variable
    ########################################################################
    for dir,sub,files in os.walk(lyricsfolder):
        for file in files:
            if songname in file:
                fin = open(lyricsfolder+os.sep+file,'r')
                flines = fin.readlines()
                fin.close()
                break
    all_lines = []
    for line in flines:
        line = line.lower()
        regex = re.compile('[,\.!?"\n]')
        stripped_line = regex.sub('', line)
        if stripped_line == '': continue
        check_for_bracket_words = stripped_line.split(' ')
        non_bracket_words = []

        for elem in check_for_bracket_words:
            if elem == "": continue #remove extra space
            if '(' in elem or ')' in elem: continue
            if elem[-1] == '\'': elem = elem.replace('\'','g') #Check if "'" is at the end of a word, then replace it with "g", eg. makin' => making
            if elem=="'cause": elem="cuz" #the ASR detects "'cause" as "cuz"
            elem=elem.replace('-',' ')
            non_bracket_words.append(elem)
        stripped_line = ' '.join(non_bracket_words)
        all_lines.append(stripped_line)
    all_lines =  ' '.join(all_lines)
    return all_lines

    #########################################################################
def CheckForNumerals(ASRtranscript_array):
    # Checks if there are numbers in a given transcript array, and converts them to words
    # eg. 16 will be converted to sixteen
    # Input: ASRtranscript_array: transcript array of words
    # Output: data: Converted/Same text depending on whether numerals are present
    ######################################################################################
    data = []
    for item in ASRtranscript_array:
        if bool(re.search(r'\d', item)):
            p = inflect.engine()
            word = p.number_to_words(item)
            data.append(word)
        else:
            data.append(item)
    return data

    #########################################################################
def GetLevenshteinScore(ASRtranscript,lyrics):
    #Creates an error matrix between transcript and lyrics using Levenshtein distance
    # Input: ASRtranscript, lyrics
    # Output: score: i.e. the error score obtained from levenshtein distance
    #         final_lyric_transcript: the resultant transcript after ASR transcript is "corrected" by the published lyrics
    #         (Refer to the paper on top of this script for me details)
    ################################################################################
    ASRtranscript_array = ASRtranscript.split(' ')
    ASRtranscript_array = CheckForNumerals(ASRtranscript_array)
    lyrics_array = lyrics.split(' ')
    error_matrix = np.zeros(shape=(len(lyrics_array)-len(ASRtranscript_array),5)) #num words in transcript by 5 types of lengths of windows on lyrics
    N = len(ASRtranscript_array)
    for lyric_index in range(len(lyrics_array)-len(ASRtranscript_array)):
        for window_ind in range(5):
            if lyric_index+N+window_ind>len(lyrics_array):
                error_matrix[lyric_index][window_ind] = np.nan
                continue
            lyric_window = lyrics_array[lyric_index:lyric_index+N+window_ind]
            num_errors,error_pattern = levenshtein(ASRtranscript_array,lyric_window)
            error_matrix[lyric_index][window_ind]=num_errors*1.0/len(lyric_window)

    min_lyric_index, min_lyric_window = np.unravel_index(np.nanargmin(error_matrix), error_matrix.shape)
    score = 1.0-np.nanmin(error_matrix)
    final_lyric_transcript = ' '.join(lyrics_array[min_lyric_index:min_lyric_index+N+min_lyric_window])
    return score, final_lyric_transcript

    #########################################################################
def GetScoreAndTranscript_LevenshteinMethod(N,lyrics,dict):
    # Create error matrix for 5 windows of words, and calculate the minimum cost transcription from the lyrics.
    # This has to happen for every ASR output
    # The ASR transcript with minimum error with its corresponding lyrics window is chosen.
    # Input: N: total number of recognition outputs
    #        lyrics: string of the published lyrics
    #        dict: the dictionary of possible transcriptions obtained from the ASR
    # Output: percent_correct[max_score_index]: the error score of the best matching transcript
    #         lyric_transcripts[max_score_index]: the resultant published lyrics of the best matching transcript
    #         ASRtranscript_array: the actual ASR transcript corresponding to the best match
    ######################################################################################
    percent_correct = [0]*N
    lyric_transcripts = []
    for i in range(N):
        transcript = dict["alternative"][i]["transcript"]
        percent_correct[i],lyric_transcript = GetLevenshteinScore(transcript.lower(),lyrics)
        lyric_transcripts.append(lyric_transcript)

    max_score_index = np.argmax(percent_correct)

    ASRtranscript = (dict["alternative"][max_score_index]["transcript"]).lower()
    ASRtranscript_array = ' '.join(CheckForNumerals(ASRtranscript.split(' ')))
    # print percent_correct[max_score_index], lyric_transcripts[max_score_index], ASRtranscript_array

    return percent_correct[max_score_index], lyric_transcripts[max_score_index], ASRtranscript_array

    #########################################################################
def RecognitionAndMatching(initialsegmentfolder,targetfile,finalsegmentfolder,songname,lyricsfolder,fout1,fout2):
    # Iterate through all the segments of that particular rendition (song)
    # For every segment, do recognition using google API
    # After recognition, do a word string alignment with the lyrics
    # If the resultant word string had more than or equal to 10 words, then the wav segment is finalized and dumped into finalsegmentfolder
    # Output files:
    ## 1. fulloutput.txt contains all valid wav files, lyric window transcript, ASR transcript, %correct
    ## 2. resultant.txt: contains the valid wav filenames whose resultant transcript has >=10 words
    ##################################################################
    ###### Capture lyrics of the song into a string variable
    lyrics = GetLyrics(lyricsfolder,songname)

    ###### Recognize the audio using Google API
    targetfile = targetfile.replace('.m4a','')
    newfile = ''
    for dir,sub,files in os.walk(initialsegmentfolder):
        for file in files:
            if targetfile in file:
                filename = file
                ## add silence at the beginning and end for better recognition
                infiles = ['silence_2seconds.wav', initialsegmentfolder + os.sep + filename, 'silence_2seconds.wav']
                data = []
                for infile in infiles:
                    w = wave.open(infile, 'rb')
                    data.append([w.getparams(), w.readframes(w.getnframes())])
                    w.close()
                newfile = initialsegmentfolder + os.sep + filename
                output = wave.open(newfile, 'w')
                output.setparams(data[0][0])
                output.writeframes(data[0][1])
                output.writeframes(data[1][1])
                output.writeframes(data[2][1])
                output.close()

                AUDIO_FILE = newfile
                print '\n' + AUDIO_FILE

                # Use the audio file as the audio source
                r = sr.Recognizer()
                with sr.AudioFile(AUDIO_FILE) as source:
                    audio = r.record(source)  # read the entire audio file

                # Recognize speech using Google Speech Recognition
                try:
                    dict = r.recognize_google(audio, show_all=True)
                    if not dict:
                        print "Google API cannot recognise this clip"
                        continue
                    N = (len(dict["alternative"]))

                    ## Algorithm for getting the best transcription
                    score, lyric_transcript, ASR_transcript = GetScoreAndTranscript_LevenshteinMethod(N,lyrics,dict)


                    # 1. fulloutput.txt contains all valid wav files, lyric window transcript, ASR transcript, %correct
                    fout1.write(filename + ',' + lyric_transcript + ','+ASR_transcript+','+str(score)+'\n')

                    # 2. resultant.txt: contains the valid wav filenames whose resultant transcript has >=10 words
                    # Transcription is that of the lyrics window
                    if len(lyric_transcript.split(' ')) >= 10:
                        fout2.write(filename + ' ' + lyric_transcript + '\n')
                        copyfile(AUDIO_FILE, finalsegmentfolder + os.sep + filename)


                except BadStatusLine:
                    print "could not fetch Google URL"
                    continue
                except sr.UnknownValueError:
                    print("Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))
                ##########################

###############################################################################################
if __name__ == "__main__":
    # This is the main function for this task
    # The input and output folders and files are already defined below
    # You will find the cleaned up subset of segments in the folder "wavsegments_final"
    # and the resultant lyrics transcriptions in "resultant.txt"
    # All the wavsegments are in wavsegments_initial, and fulloutput.txt contains complete information about all the wavsegments
    # This function runs for one .m4a file from DAMP dataset that is present in the folder "audio"
    # For the complete dataset clean up, keep all the m4a files in the "audio" folder
    #############################################################################################

    rawfolder = 'audio'# from Smule's Sing! Karaoke dataset (DAMP) here: https://ccrma.stanford.edu/damp/
    wavfolder = 'wavfiles'
    initialsegmentfolder = 'wavsegments_initial'
    finalsegmentfolder = 'wavsegments_final'
    lyricsfolder = 'lyrics'
    metadatafile = 'perfs20.csv' # Kruspe's dataset, given here: http://www.music-ir.org/mirex/wiki/2017:Automatic_Lyrics-to-Audio_Alignment

    if not os.path.exists(wavfolder):
        os.mkdir(wavfolder)
    if not os.path.exists(initialsegmentfolder):
        os.mkdir(initialsegmentfolder)
    if not os.path.exists(finalsegmentfolder):
        os.mkdir(finalsegmentfolder)

    ## output files
    fout1 = open('fulloutput.txt', 'w')
    fout2 = open('resultant.txt', 'w')


    textFromCSV = np.loadtxt(metadatafile, dtype=str, comments='#', delimiter=',')
    textFromCSV = np.delete(textFromCSV, 0, 0)

    for i in range(0,len(textFromCSV),1):
        i=231 # for the particular file in the audio folder currently, provided as an example
        print '#####################################\n','CSV index = ',i, textFromCSV[i],'\n#####################################\n'
        file = textFromCSV[i][1]+'.m4a'
        songname = textFromCSV[i][2]

        print '## convert M4A to wav, downsampling to 16kHz ##'
        rawfile = rawfolder+os.sep+file
        wavfile = wavfolder+os.sep+file.replace('.m4a',songname+'.wav')
        M4AtoWAV16k(rawfile,wavfile)

        print '## segment the wavfile to audio segments ##'
        WavSplitter(wavfile,initialsegmentfolder,file.replace('.m4a',songname+'.wav'))

        print '## segment recognition and string matching with ground truth lyrics to eliminate bad segments ##'
        RecognitionAndMatching(initialsegmentfolder,file,finalsegmentfolder,songname,lyricsfolder,fout1,fout2)

        break #remove to run for all the files
    fout1.close()
    fout2.close()



