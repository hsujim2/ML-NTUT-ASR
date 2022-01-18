# ML-NTUT-ASR
It's a homework in NTUT, for Taiwanese Speech Recognition
#environment
cpu:i5-12400
gpu:rtx 3070
system:ubuntu 21.10 LTS

#install
>sudo apt update && sudo apt upgrade
>sudo apt install python3 git vim python3-pip
>source path-to-venv/bin/activate
>pip3 install -r requirement.txt
>kaggle competitions download -c machine-learningntut-2021-autumn-asr
>unzip machine-learningntut-2021-autumn-asr.zip
>rm -r ML@NTUT-2021-Autumn-ASR/train/PexHeader

done!
#run python code
>python3 train.py

model and dictionary file will be save, then run test data
>python3 test.py

will generate or update speech_predict.csv file
#result
