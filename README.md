# OCT Volume Alignment Tool

## Overview

This program is a Graphical User Interface (GUI) application designed to load, align, and average multiple OCT (Optical Coherence Tomography) volume data sets into a single, high-quality volume. Through an intuitive interface, users can adjust processing options and monitor the progress and logs in real-time.

## Key Features

- 📁 **Batch processing** of multiple OCT volume files (`.tif`, `.tiff`)
- 🧹 **Noise removal** from the top of the image (Rows to trim)
- 🔧 **Zero-padding application** to improve alignment accuracy (Padding size)
- ⚡ **Batch processing capabilities** for large datasets (Batch size)
- 🖼️ **CLAHE contrast limit settings** to enhance en-face images (En Face Clip Limit)
- 🎯 **Two-step alignment process** using AIP (Average Intensity Projection) and en-face images
- 💾 **Automatic saving** of before/after comparison images and B-scan displacement plots

## Requirements

**Python Version**: Python 3.11 is recommended for optimal compatibility.

To run this program, the following Python libraries must be installed. You can install all of them at once using the `requirements.txt` file.

| Library | Description |
|---------|-------------|
| `numpy` | The fundamental package for scientific computing and multi-dimensional array operations |
| `opencv-python` | Used for image processing tasks like contrast enhancement and transformations |
| `tifffile` | Used for reading and writing `.tif` formatted volume data |
| `scipy` | Used for scientific and technical computing; here, it's used for phase correlation |
| `tqdm` | A library for creating smart, visual progress bars |
| `SimpleITK` | A powerful library for medical image analysis and registration, crucial for the rigid registration of volumes |
| `matplotlib` | Used to generate and save plots for visualizing results (e.g., B-scan displacements) |

## Installation and Setup

### Step 1: Clone Repository

First, clone or download this repository to a desired location on your computer. In a terminal, enter the following commands. 

```bash
git clone https://github.com/unblindness/oct_alignment.git
cd oct_alignment
```

### Step 2: Create & Activate Virtual Environment

Open a terminal in the project folder and create a virtual environment for an isolated runtime. This prevents conflicts with other Python projects on your system.

#### macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

Run the following command to install all the necessary libraries listed in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 4: Run the Program

Once all libraries are installed, run the following command to start the GUI program.

```bash
python main.py
```

**Alternative for PyCharm or other IDEs**: 
1. Open the project folder in your preferred Python IDE (PyCharm, VSCode, etc.)
2. Make sure you're using **Python 3.11** (recommended version)
3. Install the required libraries from `requirements.txt` through your IDE's package manager or terminal
4. Run `main.py` directly from the IDE interface

If successful, the **"OCT Volume Alignment Tool"** window will appear on your screen.

## How to Use

1. **Adjust Settings (Optional)**: In the 'Settings' section of the program window, you can modify the processing options. The default values work well for general use, but adjusting parameters like `Rows to trim from top` or `En Face Clip Limit` may yield better results depending on your data.

2. **Start Processing**: Click the `Start Processing` button.

3. **Select Files**: A file dialog will open, prompting you to select the OCT volume files (`.tif`, `.tiff`) you want to align. You can select multiple files at once.

4. **Select Output Folder**: After selecting your files, another dialog will open, asking you to choose a folder where the results will be saved.

5. **Monitor Progress**: Processing will begin automatically. You can monitor the overall progress in the 'Progress' section via the progress bar and status messages. The 'Processing Log' window will display detailed information for each step in real-time.

6. **Check Results**: When processing is finished, a pop-up message saying "Processing complete!" will appear. In the output folder you selected, you will find:
   - The final aligned and averaged volume file (`aligned_averaged_batch_...tif`)
   - A subfolder named `visualizations` containing:
     - `.png` images comparing the pre- and post-alignment results
     - Plots of the B-scan displacements for visual quality check of the alignment

## 📹 Video Tutorial

Watch the complete step-by-step tutorial on how to use the OCT Volume Alignment Tool:

<a href="https://youtu.be/Hb_oKiLLaqc" target="_blank">
  <img src="https://play-lh.googleusercontent.com/76AjYITcB0dI0sFqdQjNgXQxRMlDIswbp0BAU_O5Oob-73b6cqKggVlAiNXQAW5Bl1g=w600-h300-pc0xffffff-pd" alt="OCT Volume Alignment Tool Tutorial" width="600"/>
</a>

*Click the image above to watch the video tutorial on YouTube*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📚 Citation

If you use this code in your research, please cite the following paper:

```bibtex
Cellular-resolution OCT reveals layer-specific retinal mosaics and ganglion cell degeneration in vivo (under review)
by Taehoon Kim, Robby Weimer, Justin Elstrott
```

---

**Made with ❤️ for the OCT research community**