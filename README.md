# Eye Blink Detection System

A real-time eye blink detection system that monitors your blink rate and alerts you when it falls below a healthy threshold. This system is particularly optimized for users wearing glasses.

## Features

- Real-time eye blink detection using facial landmarks
- Adaptive thresholding for improved accuracy with glasses
- Blink rate monitoring and alerts
- Multiple alert methods:
  - Sound alerts
  - Popup notifications
  - Text-to-speech warnings
- Visual feedback with eye contours and statistics
- Optimized for users wearing glasses with:
  - Histogram equalization
  - Bilateral filtering
  - Adaptive thresholding
  - Relative change detection

## Requirements

- Python 3.x
- OpenCV
- dlib
- NumPy
- SciPy
- tkinter (for popup notifications)
- Git LFS (for downloading the facial landmark predictor model)

## Installation

1. Install Git LFS:

```bash
# For Ubuntu/Debian
sudo apt-get install git-lfs

# For macOS
brew install git-lfs

# For Windows (using Chocolatey)
choco install git-lfs
```

2. Clone this repository:

```bash
git clone https://github.com/yourusername/eye-blink-detection.git
cd eye-blink-detection
git lfs pull  # This will download the facial landmark predictor model
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

Note: The facial landmark predictor model (shape_predictor_68_face_landmarks.dat) will be automatically downloaded when you clone the repository using Git LFS. If you don't use Git LFS, you can manually download it:

```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

## Usage

Run the eye blink detector:

```bash
python eye_blink_detector.py
```

The system will:

1. Open your webcam
2. Detect your face and eyes
3. Monitor your blink rate
4. Alert you if your blink rate falls below 15 blinks per minute

### Controls

- Press 'q' to quit the application

### Display Information

- Blink count
- Current blink rate
- Eye aspect ratio (EAR)
- Adaptive threshold value
- Cooldown counter

## How It Works

1. **Face Detection**: Uses dlib's face detector to locate faces in the video feed
2. **Eye Landmark Detection**: Identifies 68 facial landmarks, focusing on eye regions
3. **Eye Aspect Ratio (EAR)**: Calculates the ratio of eye height to width
4. **Blink Detection**:
   - Uses adaptive thresholding based on your average eye aspect ratio
   - Detects blinks using both absolute threshold and relative changes
   - Implements a cooldown period to prevent multiple detections
5. **Alert System**: Triggers alerts when blink rate falls below the minimum threshold

## Customization

You can adjust the following parameters in the code:

- `INITIAL_BLINK_THRESHOLD`: Initial threshold for eye aspect ratio (default: 0.25)
- `MIN_BLINKS_PER_MIN`: Minimum required blinks per minute (default: 15)
- `CONSECUTIVE_FRAMES`: Frames needed to confirm a blink (default: 2)
- `BLINK_COOLDOWN`: Frames to wait before detecting next blink (default: 10)
- `ADAPTIVE_THRESHOLD_PERCENTAGE`: Percentage of average EAR for threshold (default: 0.85)

## Troubleshooting

1. **Detection Issues with Glasses**:

   - Ensure good lighting
   - Clean your glasses if needed
   - Adjust your position relative to the camera

2. **Multiple Blink Detections**:

   - The system includes a cooldown period to prevent this
   - If issues persist, adjust the `BLINK_COOLDOWN` parameter

3. **Missed Blinks**:
   - Try adjusting the `INITIAL_BLINK_THRESHOLD` or `ADAPTIVE_THRESHOLD_PERCENTAGE`
   - Ensure your face is well-lit and clearly visible

## License

This project is licensed under the MIT License - see the LICENSE file for details.
