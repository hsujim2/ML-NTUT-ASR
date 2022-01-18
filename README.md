# ML-NTUT-ASR
It's a homework in NTUT, for Taiwanese Speech Recognition

# Environment
>cpu:i5-12400<br>
>gpu:rtx 3070<br>
>system:ubuntu 21.10 LTS<br>
>python:3.9.7<br>

other software's version are in requirement.txt file

# Preparation
    sudo apt update && sudo apt upgrade
    sudo apt install python3 git vim python3-pip
    source path-to-venv/bin/activate
    pip3 install -r requirement.txt
    kaggle competitions download -c machine-learningntut-2021-autumn-asr
    unzip machine-learningntut-2021-autumn-asr.zip
    rm -r ML@NTUT-2021-Autumn-ASR/train/PexHeader
done!<br>
# Execute python code
    python3 train.py
model and dictionary file will be save, then run test data<br>

    python3 test.py
will generate or update speech_predict.csv file<br>
# Results
