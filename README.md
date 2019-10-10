# README: Automatic Sung-Lyrics Data Annotation

Created by: Chitralekha Gupta

Affiliation: NUS, Singapore

Last edited on: 12th June 2018

Last edited by: Chitralekha Gupta

This is the code base for automatically obtaining aligned lyrics for solo-singing in Smule's Sing! karaoke DAMP dataset. It provides a cleaner annotated dataset.

Please refer to the following paper for details:

Gupta, C., Tong R., Li, H. and Wang, Y., 2018, September, "SEMI-SUPERVISED LYRICS AND SOLO-SINGING ALIGNMENT". Accepted for ISMIR 2018, Paris.(http://ismir2018.ircam.fr/doc/pdfs/30_Paper.pdf)

## Contents
This consists of the following:
- Audio folder (folder: audio), where an example audio (.m4a) from DAMP dataset is provided
- Converted .m4a to .wav are in the folder: wavfiles
- All ~10 seconds segments are in the folder: wavsegments_initial
- Cleaned up subset of segments are in the folder wavsegments_final, the corresponding automatically obtained lyrics annotations are in "resultant.txt"
- fulloutput.txt contains all segment names, lyric window transcript, ASR transcript, and %correct
- Lyrics folder (folder: lyrics), where lyrics are extracted from Smule Sing! website
- 'perfs20.csv' is a meta-data file from Kruspe's dataset, given here: http://www.music-ir.org/mirex/wiki/2017:Automatic_Lyrics-to-Audio_Alignment
- Rest are python scripts and other dependency files.


## Dependencies
- This program is designed for monophonic (without background music) audio files.

## How to run?
- The python scripts main.py is the main file.  
- Start with main.py, and follow the instructions in the file header. 
- The script currently runs for one example file present in the folder "audio"
- You can add more files from the DAMP dataset to the "audio" folder

## License
The code and models in this repository are licensed under the GNU General Public License Version 3. For commercial use of this code and models, separate commercial licensing is also available. Please see the contacts below.

## Contact
- Chitralekha Gupta: chitralekha[at]u[dot]nus[dot]edu
- Haizhou Li: haizhou[dot]li[at]nus[dot]edu[dot]sg
- Ye Wang: wangye[at]comp[dot]nus[dot]edu[dot]sg
