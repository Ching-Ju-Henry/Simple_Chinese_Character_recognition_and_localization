# Simple Chinese Character recognition and localization

## Overview
The goal of this project is to recognize and localize the multiple chinese characters from one to ten on an image(from webcam), and then we can change their font from our "font" dataset. We collect some hand-writing and computer font chinese characters from one to ten in order to train a SVM model to recognize characters. Besides, we also design an algorithm to localize the characters position. Then we can use above to implement Simple Chinese Character recognition and localization. 

## Demo
### Dataset
1. "data" folder：Including hand-writing and computer font chinese characters from one to ten to train SVM
2. "font" folder：Including 10 different computer font

### Training
1. Using *hog.py* to collect all the chinese characters feature in the "data".
2. Using *svm.py* to train a model in order to recognize characters.

### Testing
1. Using *main.py* to testing the image from webcam, and then our algorithm can change the chinese from input image.
&emsp;**[Note]**
"function_ver2.py" is used for localization 


## Result
**1. Flow Chart**
<center>
<img src="./Flow/flow.jpg" >
<br>
</center>

**2. Some Results**
<table border=1>
<tr>
<td>
<img src="./resize/TV_resize.jpg" width="20%"/>
</td>
</tr>

</table>

## Requirements
* 

