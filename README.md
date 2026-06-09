# SAM2 Eye Segmentation Pipeline

An interactive, modular Python pipeline designed to perform eye structure segmentation (e.g., iris, pupil, conjunctiva) from video sequences. It incorporates an interactive manual template classification step, propagates states using ORB feature matching, and handles automated deep-learning mask generation and propagation using Meta's Segment Anything 2 (SAM2) Video Predictor.

---

## Architecture Overview

The system is split into specialized modules to handle data isolation, interactive UI orchestration, and deep learning inference tracking:

* **`main.py`**: Execution entry point. Manages argparsing and sequential module coordination.
* **`config.py`**: Global configuration file handling parameters, environment variables, paths, and keybindings.
* **`modules/frame_manager.py`**: Indexing layer encapsulating operations over frame sequence images saved on disk.
* **`modules/frame_extractor.py`**: Handles video streaming decoding, downsampling, and vertical central splitting (50/50) into Left and Right sequence subsets.
* **`modules/classifier/`**: Contains the interactive `ClassifierUI`, `ClassificationStorage` cache, and `ORBClassifier` feature matcher used to distinguish between open and closed eye frames.
* **`modules/sam/`**: Integrates SAM2 Image Predictor (`ManualMaskCreator`), memory-efficient tracking over large frame steps (`BatchProcessor`), validation schema definitions (`MaskValidator`), and full workflow coordination (`SAMVideoInference`).
* **`modules/storage/`**: State management layer handling structured file routing (`FolderManager`), operational JSON sessions tracking (`SessionManager`), and spreadsheet serialization data logging (`CSVManager`).
* **`modules/ui/`**: Pure UI wrappers handling multi-viewport evaluation rendering canvases (`MaskViewer`) and basic key interaction abstractions (`VideoPlayer`, `KeyboardController`).

---

## Setup & Installation

### 1. System Requirements & Core Dependencies
Ensure you have Python 3.10+ installed on your system. This pipeline relies on **PyTorch (with CUDA support recommended for GPU acceleration)** and **OpenCV**.

- Linux with Python ≥ 3.10, PyTorch ≥ 2.5.1 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation. Install them together at https://pytorch.org to ensure this.
  * Note older versions of Python or PyTorch may also work. However, the versions above are strongly recommended to provide all features such as `torch.compile`.
- [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) that match the CUDA version for your PyTorch installation. This should typically be CUDA 12.1 if you follow the default installation command.
- If you are installing on Windows, it's strongly recommended to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu.


### 2. Platform-Specific Installation Commands

Choose the instructions matching your operating system below:

#### MacOS
```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch (CPU or MPS acceleration optimized default)
pip install torch torchvision

# 3. Install remaining pipeline dependencies
pip install -r requirements.txt

# 4. Install SAM2 repository from source
pip install git+[https://github.com/facebookresearch/sam2.git](https://github.com/facebookresearch/sam2.git)
```

#### Linux
```bash
# 1. Install system prerequisites for OpenCV if missing
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 4. Install remaining pipeline dependencies
pip install -r requirements.txt

# 5. Install SAM2 repository from source
pip install git+[https://github.com/facebookresearch/sam2.git](https://github.com/facebookresearch/sam2.git)
```

### 3. Model Weights Download

Download your preferred SAM2 checkpoint model weight and match its storage location framework parameters inside your local shell environment or modify them within `config.py`:

``` bash
# Example downloading the base model weights
wget [https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
export SAM2_CHECKPOINT="/path/to/sam2.1_hiera_base_plus.pt"
```

## Usage Guide

Modify `SAM2_CHECKPOINT` path in `config.py` with the correct path to `sam2.1_b.pt` file.

Run the pipeline from your command terminal executing `main.py` along with the positional target parameters:
```bash
python3 main.py <video_path> <object> <start_frame> <end_frame> <eye>
```

### Argument Definitions

- `video_name`: Name of the target video file inside the workspace (without file extension tokens).

- `object`: Targeted anatomical eye element to classify and segment (iris | pupil | conjunctiva).

- `start_frame`: Lower processing constraint index boundary (0 begins from the video origin).

- `end_frame`: Upper processing constraint index boundary (-1 runs until the video stream terminates).

- `eye`: Targeted structural split tracking mode parameters (left | right | both).

### Practical Execution Examples
```bash
# Process the iris structure from the beginning to end for the right eye split sequence
python3 main.py video_34 iris 0 -1 right

# Track the pupil structure over a specific frame window range on the left side sequence
python3 main.py video_34 pupil 1000 5000 left

# Extract and process both eyes simultaneously on separate consecutive loops
python3 main.py video_34 conjunctiva 0 -1 both
```

## Interactive Controls
### 1. Interactive Eye State Classifier Window (`ClassifierUI`)
During execution, a downsampled sequence viewport helps manually train template states:

- `o`: Lables the active displayed frame under the `OPEN` state category.
- `c`: Labels the active displayed frame under the `CLOSE` state category.
- `b`: Opens/Closes a range grouping selector `BATCH MODE` block to apply a status globally over a frame interval.
- `n`: Skip the active frame template generation step without assigning labels.
- `r`: Clears out active storage dictionaries to `RESET / RESTART` the manual task from 0.
- `ESC`: Aborts session execution loops immediately without exporting data.

### 2. Multi-Mask Inference Previewer (`MaskViewer`) *IN DEVELOPMENT*
When inference propagation terminates, an optional preview screen opens:

- `← / →`: Decrement or Increment manual step frames index positions.
- `SPACEBAR`: Toggles automatic constant sequential stream playback preview configurations.
- `m`: Cycles viewing rendering layouts:`Overlay Blended Mask` $\rightarrow$ `Raw Binary Mask` $\rightarrow$ `Side-by-Side Comparison Matrix`.
- `s`: Captures and exports a localized .png image screenshot directly inside visualizations outputs storage folders.
- `ESC`: Closes out visualization viewports safely.

## Outputs & Data Structure

The pipeline completely automates directory management through the `FolderManager` module. All generated assets, binary tracking masks, and session logs are saved under a unified root directory named after the source video sequence.

### Output Directory Tree

Upon processing a video (e.g., `video_34`), the system populates the following structured workspace:

```text
.
└── video_34/
    ├── sessions/
    │   └── session.json
    ├── left/
    │   ├── frames/
    │   │   ├── video_34_frame_0000000_L.png
    │   │   └── ...
    │   ├── video_34_left_classification.csv
    │   ├── masks/
    │   │   └── iris/
    │   │       ├── initial_mask.png
    │   │       ├── last_mask.npy
    │   │       ├── mask_video_34_frame_0000000_L.png
    │   │       └── ...
    │   └── visualizations/
    │       └── iris/
    │           ├── final_video_34_frame_0000000_L.png
    │           └── ...
    └── right/
        ├── frames/
        │   ├── video_34_frame_0000000_R.png
        │   └── ...
        ├── video_34_right_classification.csv
        ├── masks/
        │   └── <object>/
        └── visualizations/
            └── <object>/
```
### Artifact Descriptions
#### 1. Session Logs `sessions/`

- **session.json**: A tracking file managed by `SessionManager`. It stores execution states across runs to easily resume work. It logs whether open/close classification has been completed, lists processed structures, and references the exact integer frame checkpoint where the last deep-learning inference loop was stopped.

#### 2. Extracted Frames `<eye>/frames/`

- Contains the raw `.png` image files generated by splitting the original source video down the vertical center. Left eye assets are appended with the `_L.png` suffix, and right eye assets with `_R.png`. Frame tracking IDs are formatted using a fixed 7-digit zero-padded naming sequence (`0000000`).

#### 3. State Propagation Records `<eye>/_classification.csv`

- A comma-separated index sheet mapping every extracted file to its calculated eye-state structure. It features three headers: `frame_filename`, `frame_id`, and `state` (`open` | `close`). Open frames denote valid windows where SAM2 tracks the requested anatomical structure, whereas closed frames immediately yield empty blank output masks.

#### 4. Binary Tracking Masks `<eye>/masks/<object>/`

- `initial_mask.png`: The static baseline reference mask annotated manually by the user during the initialization stage.

- `last_mask.npy`: A serialized float32 NumPy matrix array that acts as a rolling memory checkpoint between individual batch blocks. It allows the system to seamlessly flush GPU VRAM without breaking tracking continuity.

- `mask_*.png`: The final output single-channel binary masks (thresholded grayscale images where pixel values are strictly `0` for background or `255` for the targeted segmented object).

#### 5. Visualization Overlays `<eye>/visualizations/<object>/`

- `final_*.png`: High-resolution composite preview images where the output mask is blended over the original raw frame using a semi-transparent alpha layer (colored green by default). This folder serves as the target image database directory scanned by the interactive `MaskViewer` tool.
