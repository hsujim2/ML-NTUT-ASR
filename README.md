# ML-NTUT-ASR
It's a homework in NTUT, for Taiwanese Speech Recognition
# environment
cpu:i5-12400<br>
gpu:rtx 3070<br>
system:ubuntu 21.10 LTS<br>
# install
>sudo apt update && sudo apt upgrade<br>
>sudo apt install python3 git vim python3-pip<br>
>source path-to-venv/bin/activate<br>
>pip3 install -r requirement.txt<br>
>kaggle competitions download -c machine-learningntut-2021-autumn-asr<br>
>unzip machine-learningntut-2021-autumn-asr.zip<br>
>rm -r ML@NTUT-2021-Autumn-ASR/train/PexHeader<br>

done!<br>
# run python code
>python3 train.py<br>

model and dictionary file will be save, then run test data<br>
>python3 test.py<br>

will generate or update speech_predict.csv file<br>
# result
