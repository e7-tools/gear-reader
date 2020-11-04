# E7 Gear Reader
A tool to import inventory from Epic Seven Mobile Game

This repo contain the source code of the gear reader component 
from [e7gears.herokuapp.com](https://e7gears.herokuapp.com). 
It also includes a simple GUI to review and edit imported inventory.

The program take a recording in mp4 format as input. 
Please visit the website above for detail requirements for the recording.

![Screenshot](/templates/reader-ss.png)

### Setup
Install Tesseract-OCR 4. Instructions can be found [here](https://github.com/tesseract-ocr/tesseract).

Change TESSERACT_DIR variable in config.py to your file. i.e. 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

Run reader.py to open the interface.


### Issues
Users are able to edit value of a substat (i.e change Speed from 11 to 17) but **cannot** edit substat name (i.e change Speed 11 to Attack 11). 
*Luckily, errors in substat are rare.*