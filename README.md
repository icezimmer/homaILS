# homaILS

**homaILS** is a Python-based project focused on [provide a brief description of your project's purpose, e.g., "implementing indoor localization systems using sensor fusion techniques"].

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/icezimmer/homaILS.git
   cd homaILS
   ```

2. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the main calibration task, execute:

```bash
python calibration_task.py
```

For positioning tests, you can run:

```bash
python positioning_test.py
```

Replace `positioning_test.py` with other test scripts as needed, such as `positioning_test_19.py`, `positioning_test_56.py`, or `positioning_test_outdoor.py`.

**Note**: Ensure that the necessary data files are present in the `data` directory before running the scripts.

## Project Structure

```
homaILS/
├── data/
│   ├── [data files]
├── homaILS/
│   ├── __init__.py
│   ├── [module files]
├── .gitignore
├── README.md
├── calibration_task.py
├── cov_process_noise.py
├── positioning_test.py
├── positioning_test_19.py
├── positioning_test_56.py
├── positioning_test_outdoor.py
├── requirements.txt
```

- `data/`: Contains datasets and related files.
- `homaILS/`: Core module with initialization and other Python files.
- `calibration_task.py`: Script for calibration tasks.
- `cov_process_noise.py`: Script related to process noise covariance.
- `positioning_test*.py`: Scripts for various positioning tests.
- `requirements.txt`: List of Python dependencies.

## Features

- Calibration routines for sensor data.
- Multiple positioning test scenarios, including outdoor environments.
- Modular code structure for easy extension and maintenance.
