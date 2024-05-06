# README for RAGECAN: Automatic Gene Expression Reduction for Cancer Classification

## Introduction

RAGECAN is a Python-based framework designed specifically for improving cancer classification tasks through gene expression data. Utilizing a suite of 10 different dimensionality reduction methods, RAGECAN enhances the performance of classifiers and streamlines the process of parameter optimization and reduction technique application. This framework features an intuitive graphical user interface (GUI) aimed at making advanced data analysis techniques accessible to researchers without a background in computational programming. With built-in functionalities for result tracking and validation, RAGECAN ensures the integrity and reliability of the metrics obtained.

## Features

- **10 Dimensional Reduction Methods**: A comprehensive collection of techniques to optimize gene expression data analysis.
- **Automatic Parameter Optimization**: Simplifies the tuning of both dimensionality reduction methods and classifiers.
- **User-Friendly GUI**: Facilitates the use of sophisticated algorithms through a straightforward graphical interface.
- **Result Tracking and Validation**: Ensures the accuracy and reliability of analysis results through robust validation mechanisms.

## Installation

To get started with RAGECAN, follow these steps to set up the environment and run the application:

### Prerequisites

Ensure you have Python 3.9 installed on your system. If not, you can download it from the official [Python website](https://www.python.org/downloads/).

### Step-by-Step Installation

1. **Set up Python Environment**:
   - Install and set up a Python 3.9 environment on your system. You can use virtual environments like `venv` for a localized setup:
     ```bash
     python3 -m venv ragecan-env
     source ragecan-env/bin/activate  # On Windows use `ragecan-env\Scripts\activate`
     ```

2. **Install Dependencies**:
   - Navigate to the root folder of RAGECAN and install the required Python libraries:
     ```bash
     pip install -r requirements.txt
     ```

3. **Launch MLflow**:
   - To track experiments, start the MLflow server by opening a terminal and executing:
     ```bash
     mlflow server --host 127.0.0.1 --port 5600
     ```

4. **Run the Application**:
   - In another terminal, start the RAGECAN application:
     ```bash
     python app.py
     ```

5. **Access the GUI**:
   - Open a web browser and access the RAGECAN GUI at:
     ```
     http://localhost:5000
     ```
   - Upload your dataset, configure your experiment, and start the analysis!

## Usage

After setting up the RAGECAN framework as described, you can begin using the GUI to perform gene expression data analysis. The interface allows for easy upload of data, setting up of experiments, and execution of analysis, all through a few clicks.

## Conclusion

RAGECAN is designed to empower researchers by simplifying complex data analysis tasks through automation and user-friendly interfaces, ultimately aiming to enhance the classification of various types of cancer through precise and reliable gene expression data analysis.

For further information or to report issues, please refer to the GitHub repository or contact me.