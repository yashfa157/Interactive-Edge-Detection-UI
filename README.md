# Interactive Edge Detection UI

An interactive web-based application built using **Python**, **OpenCV**, and **Streamlit** that allows users to experiment with different **edge detection algorithms**.  
Users can upload an image, select an algorithm (Sobel, Laplacian, or Canny), adjust its parameters, and instantly visualize the results side by side with the original image.

---

## Project Description

This project provides a hands-on visual understanding of how edge detection algorithms work and how their parameters affect the results.  
It is designed for educational and experimental use in computer vision courses.

---

## Setup and Installation

### 1. Clone the repository
```bash
git clone https://github.com/yashfa157/Interactive-Edge-Detection-UI.git
cd Interactive-Edge-Detection-UI
```

### 2. Create and activate a virtual environment

## Windows:

```bash
python -m venv venv
venv\Scripts\activate
```
## Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install required dependencies
```bash
pip install -r requirements.txt
```

Once dependencies are installed, start the Streamlit app with:
```bash
streamlit run edge_detection_ui.py
```
Then open the local URL shown in the terminal.

### Features Implemented

Image Upload: Supports JPG, PNG, and BMP formats.
Edge Detection Algorithms:
Sobel – adjustable kernel size and gradient direction (X, Y, or Both)
Laplacian – adjustable kernel size
Canny – adjustable thresholds, Gaussian kernel size, and sigma
Parameter Adjustment UI: Uses sliders, dropdowns, and radio buttons for easy tuning.
Real-time Updates: Output refreshes instantly when parameters change.
Side-by-side Display: Original and processed images are shown side by side.
Clean & Responsive UI: Designed with Streamlit for an intuitive user experience.

## Screenshots


Below are the visual results of each edge detection algorithm.
Each image shows the original input (left) and the edge-detected output (right) side by side.

### Sobel Edge Detection
![Sobel Result](Outputs/Sobel.png)
![Sobel Result](Outputs/Sobel(2).png)
![Sobel Result](Outputs/Sobel(3).png)
![Sobel Result](Outputs/"Sobel y direction".png)

### Laplacian Edge Detection
![Laplacian Result](Outputs/Laplacian.png)
![Laplacian Result](Outputs/Laplacian(2).png)

### Canny Edge Detection
![Canny Result](Outputs/Canny.png)
![Canny Result](Outputs/Canny(2).png)
![Canny Result](Outputs/Canny(3).png)

### Author

Name: Yashfa Najam
Course: Computer Vision
Project: Edge Detection Algorithms Visualization
GitHub Repo: Interactive Edge Detection UI

### License

This project is developed for academic purposes and is free to use for learning and educational demonstrations.