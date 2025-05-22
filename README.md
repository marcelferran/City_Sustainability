# Urban Monitoring System with Deep Learning

**`city_sustainability` - Bootcamp_project**

A deep learning-based system to classify urban land occupancy using satellite imagery and calculate quality of life metrics to support urban sustainability.

## 🧩 Introduction

Urban sustainability is a fundamental pillar for achieving net-zero carbon emissions and effectively addressing the inherent challenges of population growth and climate change. In this context, emerging Fourth Industrial Revolution technologies, such as deep learning and geospatial analysis, are positioned as disruptive tools that open new and limitless possibilities for precise monitoring and substantial improvement of quality of life in urban environments.

## 🔧 Project Development

### Objective

The main objective of this project is to accurately identify and classify urban land occupancy, and then derive and calculate relevant quality of life metrics from high-resolution satellite and aerial imagery.

### Workflow

The system operates under a structured workflow:

1.  **Input:** High-resolution satellite or aerial images are used as the primary data source.
2.  **Model:** A U-Net convolutional neural network, strengthened with the VGG-16 architecture as its base, processes the images.
3.  **Output:** The result is a segmented image, where each pixel is classified into one of nine land occupancy classes (e.g., buildings, vegetation, roads, agriculture, water bodies, vacant lots, etc.).
4.  **Classifier:** Subsequently, a classifier assigns quality of life levels (High, Medium, Low) using a set of multi-factorial metrics encompassing environmental, infrastructure, and land use aspects.

### Data Used

* **Source:** High-resolution images from 44 countries, obtained from the Open Earth Map initiative.
* **Labeling:** Meticulous manual labeling of images was performed across 9 distinct land occupancy classes.

### Model Architecture

The core of the architecture lies in a **U-Net**, enhanced by the implementation of **VGG-16** for knowledge transfer. This configuration allows for:

* **VGG-16:** Leverages pre-trained high-level features, accelerating learning and improving robustness.
* **U-Net:** Facilitates detailed and precise segmentation through its symmetric encoding and decoding progressive feature architecture, ideal for semantic image segmentation tasks.
* **Training:** The model was specifically trained to recognize spatial and semantic patterns intrinsic to the urban sustainability context.

### Results

The results obtained during model training and validation demonstrate significant improvement:

* **Accuracy:** Improved from 0.22 to **0.60**.
* **IOU (Intersection over Union):** Increased from 0.12 to **0.31**.
* **Performance:** Good performance was observed in key categories such as `buildings` and `agriculture`.
* **Limitations:** Areas for improvement were identified, particularly in the precise classification of `vacant lots`.

## 📊 Conclusions

The development of this system validates the feasibility and effectiveness of using deep neural networks for urban image segmentation and the subsequent derivation of quality of life metrics. The generated metrics offer invaluable capabilities for evaluating urban environments from a holistic perspective (environmental, infrastructural, and territorial), which in turn facilitates informed, data-driven decision-making to drive more sustainable urban development.

## 🚀 Way Forward (Next Steps)

To continue evolving this project and maximize its impact, the following steps are proposed:

* **Expand Dataset:** Increase the diversity and volume of training data to improve model generalization and robustness across different urban contexts.
* **Optimize Model Architecture:** Investigate and apply architectural optimizations to achieve greater accuracy, especially in classifying challenging classes such as `vacant lots`.
* **Integration with Urban Dashboards:** Develop integration modules that allow quality of life metrics to be visualized in real-time on interactive dashboards, offering timely analysis to local authorities and urban planners.
* **Develop a Public API:** Create a publicly accessible Application Programming Interface (API) to facilitate programmatic access to sustainability metrics for developers, urban planners, researchers, and interested citizens.

## ⚙️ Installation and Usage Instructions

### Virtual Environment Setup (pyenv)

It is highly recommended to use `pyenv` for isolated Python environment management.

1.  **Create the virtual environment:**
    ```bash
    pyenv virtualenv city_sustainability
    ```
2.  **Activate the virtual environment locally:**
    ```bash
    pyenv local city_sustainability
    ```

### Dependency Installation

1.  **Install packages from `requirments.txt`:**
    ```bash
    pip install -r requirments.txt
    ```
    *Note: Ensure the file name is `requirments.txt` and not `requirements.txt` if that is your case.*

2.  **Install the `city_sustainability` package in editable mode:**
    This allows changes in the code to be reflected without needing reinstallation.
    ```bash
    pip install -e .
    ```

### Project Package Usage

Once installed, you can import and use the project modules:

* **For Data Preprocessing:**
    ```python
    from city_sustainability.preprocessing import image_resize
    ```

* **For Data Loading:**
    ```python
    from city_sustainability.loading_data import loading_paths, image_and_label_arrays
    ```

* **For Modeling (U-Net):**
    ```python
    from city_sustainability.models import unet_model

    # Create the model:
    # 'a', 'b', and 'c' should be replaced with the correct values for input_shape and num_classes
    model = unet_model.build_model(input_shape=(a, b, 3), num_classes=c)

    # Compile the model:
    unet_model.compile_model(model)

    # Train the model:
    # 'x_train' and 'y_train' should be your training data
    history = unet_model.train_model(model, x_train, y_train, epochs=1, batch_size=512, validation_split=0.2)

    # Evaluate the model:
    # 'x_test' and 'y_test' should be your test data
    unet_model.evaluate_model(model, x_test, y_test)

    # Make predictions:
    # 'x_test' should be your data for prediction
    predictions = unet_model.predict(model, x_test)
    ```

### Streamlit Application Usage

The project includes an interactive interface developed with Streamlit.

1.  **Install Streamlit:** (Should already be included if you ran `pip install -r requirments.txt`).
    Ensure you are in your project's root directory and that your `requirments.txt` contains `streamlit`.

2.  **Navigate to the Streamlit folder and run the application:**
    First, ensure you are in the `city_sustainability/streamlit` directory if your `Home.py` is located there, or in the root of `city_sustainability` if `Home.py` is at the root.
    ```bash
    # Assuming Home.py is inside a folder named 'streamlit' in your project's root
    cd streamlit
    streamlit run Home.py
    ```
    If `Home.py` is in the root of your `city_sustainability` project, simply run:
    ```bash
    streamlit run Home.py
    ```

### Jupyter Notebook/JupyterLab Verification

To ensure your environment and notebooks are working correctly, you can try starting Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
# or
jupyter lab

### END
