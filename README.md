# Velociraptor: ARCore-Based 3D Point Tracking and Depth Estimation

A comprehensive system for real-time 3D point tracking and depth estimation using ARCore on Android devices, combined with advanced Python-based analysis tools for trajectory reconstruction and depth model evaluation.

## Overview

Velociraptor is a research project that combines mobile AR technology with computer vision to enable precise 3D point tracking and depth estimation. The system consists of two main components:

1. **Android Application** (`velocity_app/`): Real-time data collection using ARCore
2. **Python Analysis Pipeline** (`python/`): Offline processing and trajectory reconstruction

## Features

### Android Application
- **ARCore Integration**: Uses ARCore's raw depth API for real-time depth sensing
- **Point Tracking**: Optical flow-based tracking of user-selected points across frames
- **Depth Estimation**: TensorFlow Lite integration with MiDaS depth estimation models
- **Data Collection**: Batch collection of synchronized camera frames, depth data, and tracking information
- **Real-time Visualization**: Live preview with depth point overlay and tracked point visualization

### Python Analysis Pipeline
- **Multi-Model Depth Estimation**: Support for various depth estimation models (MiDaS, Depth Anything)
- **3D Trajectory Reconstruction**: Converts 2D tracked points to 3D trajectories using camera intrinsics and extrinsics
- **Batch Processing**: Automated processing of collected data batches
- **Visualization Tools**: Comprehensive plotting and analysis of trajectories and depth maps
- **Regression Analysis**: Statistical analysis of tracking accuracy and depth estimation performance

## Project Structure

```
Velociraptor-app/
├── velocity_app/                 # Android application
│   ├── src/main/java/           # Java source code
│   │   └── com/google/ar/core/velociraptor/
│   │       ├── rawdepth/        # Main activity and frame processing
│   │       └── common/          # Helper utilities
│   └── build.gradle             # Android build configuration
├── python/                      # Python analysis scripts
│   ├── A*_files_*.py           # File management utilities
│   ├── B*_build_*.py           # Data processing and model building
│   ├── B*_display_*.py         # Visualization scripts
│   ├── C*_display_*.py         # Advanced visualization
│   ├── D*_witness_*.py         # Camera witness tools
│   ├── L*_lib_*.py             # Core library functions
│   └── persistence_helpers.py  # Data persistence utilities
├── MiDaS/                      # Depth estimation models
│   ├── midas/                  # Model implementations
│   ├── weights/                # Pre-trained model weights
│   └── README.md               # MiDaS documentation
├── MiDaS_weights/              # Additional model weights
├── opencv/                     # OpenCV Android module
├── demos/                      # Example videos
└── exported/                   # Data export directory
```

## Requirements

### Android Application
- Android Studio 4.0+
- Android SDK 24+ (API level 24+)
- ARCore-compatible device
- OpenCV for Android
- TensorFlow Lite

### Python Analysis
- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Pandas
- Transformers (for Depth Anything models)

## Installation

### Android Application

1. Clone the repository:
```bash
git clone <repository-url>
cd Velociraptor-app
```

2. Open the project in Android Studio
3. Ensure you have the required SDK versions installed
4. Build and install the APK on an ARCore-compatible device

### Python Environment

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install torch torchvision
pip install opencv-python
pip install numpy matplotlib pandas
pip install transformers
```

3. Download MiDaS model weights to `MiDaS_weights/` directory

## Usage

### Data Collection (Android)

1. Launch the Velociraptor app on your ARCore device
2. Grant camera permissions when prompted
3. Tap on the screen to select a point to track
4. Press "Collect Frames" to start data collection
5. The app will collect 20 frames per batch with synchronized:
   - Camera images
   - ARCore depth points
   - Depth estimation model outputs
   - Tracked point coordinates
   - Camera intrinsics and extrinsics
   - Timestamps

### Data Analysis (Python)

1. Transfer exported data from device to your analysis machine
2. Configure file paths in the Python scripts:
```python
FILE_PATH = "path/to/exported/data"
```

3. Run analysis scripts in sequence:
```bash
# Display collected images and depth points
python B0_display_images.py

# Build depth estimation models
python B1_build_model_depth_batch.py

# Build 3D trajectories
python B5_build_trajectories_3D.py

# Display trajectory analysis
python C1_display_trajectories_and_vel_3D.py
```

## Key Components

### Android Application (`VelociraptorActivity.java`)

The main activity handles:
- ARCore session management
- Real-time camera preview with depth overlay
- Point tracking using OpenCV optical flow
- TensorFlow Lite model inference for depth estimation
- Batch data collection and export

### Python Analysis Pipeline

- **L1_lib_extraction_and_visualisation.py**: Core utilities for data extraction and visualization
- **B1_build_model_depth_batch.py**: Depth estimation model processing
- **B5_build_trajectories_3D.py**: 3D trajectory reconstruction
- **C1_display_trajectories_and_vel_3D.py**: Trajectory analysis and visualization

## Data Format

The system exports data in binary format with the following structure:
- `batch_X_timestamps_Y.bin`: Frame timestamps
- `batch_X_depth_points_Y.bin`: ARCore depth points
- `batch_X_depth_map_camera_Y.bin`: Camera images
- `batch_X_depth_map_colour_Y.bin`: Color-mapped depth estimates
- `batch_X_tracked_point_Y.bin`: Tracked point coordinates
- `batch_X_camera_intrinsics_Y.bin`: Camera calibration data
- `batch_X_extrinsic_matrix_Y.bin`: Camera pose matrices

## Research Applications

This system is designed for research in:
- Mobile AR and computer vision
- Depth estimation accuracy evaluation
- 3D trajectory tracking
- Multi-modal sensor fusion
- Real-time point tracking algorithms

## Contributing

Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the Apache License 2.0 - see the `LICENSE` file for details.

## Acknowledgments

- ARCore team for the depth sensing API
- MiDaS project for depth estimation models
- OpenCV community for computer vision tools
- TensorFlow Lite for mobile inference

## Demo Videos

Check the `demos/` directory for example videos showing:
- Point tracking with depth visualization
- Trajectory reconstruction results

## Troubleshooting

### Common Issues

1. **ARCore not supported**: Ensure your device supports ARCore and has the latest version installed
2. **Depth data not available**: Some devices may not support raw depth API
3. **Model loading errors**: Verify TensorFlow Lite model files are in the correct location
4. **Python import errors**: Ensure all dependencies are installed in the correct environment

### Performance Tips

- Use smaller depth estimation models for better real-time performance
- Adjust frame collection rate based on device capabilities
- Process data in smaller batches for memory efficiency

## Citation

If you use this project in your research, please cite:

```bibtex
@software{velociraptor2024,
  title={Velociraptor: ARCore-Based 3D Point Tracking and Depth Estimation},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-username]/Velociraptor-app}
}
```
