# Coins & Screws Counter and Sorter

| | |
|---|---|
| **Course** | CS464 - Computer Vision |
| **Instructor** | Dr. Khaled Al Bahnasi |
| **University** | JIC |
| **Student** | Ahmed Zaitoun |
| **Section** | 5 |
| **Student ID** | 231100182 |
| **Email** | ahmed.zaitoun.ps@gmail.com |

Count and classify coins and screws with classical computer vision (OpenCV, no model training),
with a Streamlit interface to upload images and analyse them.

## Pipeline
1. **Preprocessing & thresholding** - grayscale, Gaussian blur, Otsu threshold.
2. **Morphology & Watershed** - opening/closing, hole filling, Distance Transform + Watershed to split touching coins.
3. **Shape features** - circularity (4πA/P²) and aspect ratio separate round coins from elongated screws.
4. **Coin classification** - diameter in mm via PX_PER_MM calibration + HSV colour of centre vs outer ring.
5. **Screw classification** - length from minAreaRect, grouped into short / medium / long.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
