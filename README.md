---
title: Chatterbox-Multilingual-TTS
emoji: 🌎
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
short_description: Chatterbox TTS supporting 23 languages
---


windows
[1]  py -3.10 -m venv venv


[2] venv/Scripts/activate

[3] ip install --upgrade setuptoolsp

[4] C:\Users\admin\projects\python-project\text-to-voice\chatterbox\venv\Scripts\python.exe -m pip install --upgrade pip

[5] pip install --upgrade setuptools wheel

[6] pip install -e .

[7] python Run.py


ubuntu pyhton 3.12 for colab



!git clone https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS
%cd Chatterbox-Multilingual-TTS
!pip install -r requirements.txt
!pip install spaces
!python Run.py


document.querySelector("#cell-ZDvAuIRE01Nh > div.main-content > div > div.codecell-input-output > div.inputarea.horizontal.layout.code > div.cell-gutter > div > colab-run-button").shadowRoot.querySelector("#stopSymbolMask")