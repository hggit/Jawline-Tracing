# Jawline Tracing
### Detect human faces and trace jawline using facial landmarks

Dependencies(Use python 3+):
- `numpy`
- `dlib`
- `opencv-python`

Clone repo and run script as

    $ python landmark_jaw.py path/to/image.jpg

The generated file `output.txt` contains the 68 landmark co-ordinates for each face detected and the image `output_img.jpg` has the jawline curve on the input image.
