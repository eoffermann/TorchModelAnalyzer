# TorchModelAnalyzer

TorchModelAnalyzer is a Python utility script for analyzing PyTorch `.pth` model files. It provides a detailed JSON report about the contents of the `.pth` file, including the structure, metadata, and parameter details.

## Features

- Identifies common keys in `.pth` files such as `state_dict`, `optimizer`, `params`, and `params_ema`.
- Analyzes model parameters, including shapes and total parameter count.
- Provides a summary of the `.pth` file contents.
- Extracts module hierarchy and details for saved PyTorch models.

---

## Setup

This project requires Python 3.10.11 and Conda for environment management.

### Step 1: Create a Conda Environment
Run the following commands to create and activate a Python environment compatible with the script:

```bash
conda create -n torch_model_analyzer python=3.10.11 -y
conda activate torch_model_analyzer
```

### Step 2: Install Dependencies
Install the required dependencies using `pip` and the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Usage

### Analyze a `.pth` File

The script, `analyze.py`, analyzes the contents of a `.pth` file and outputs a detailed JSON description.

#### Basic Usage
Run the script with the path to the `.pth` file:

```bash
python analyze.py /path/to/your_model.pth
```

#### Save Output to a File
Optionally, save the JSON output to a file using the `--output` argument:

```bash
python analyze.py /path/to/your_model.pth --output /path/to/output.json
```

---

## Example Outputs

### Input

Given a `.pth` file located at `/path/to/your_model.pth`, run the script:

```bash
python analyze.py /path/to/your_model.pth
```

### Output

The script will output a detailed JSON description. Below is an example of what you might expect:

```json
{
    "file_path": "/path/to/your_model.pth",
    "keys": [
        "params"
    ],
    "details": {
        "state_dict": "Not found",
        "optimizer": "Not found",
        "params": {
            "num_parameters": 1234567,
            "parameter_shapes": {
                "conv1.weight": [64, 3, 3, 3],
                "conv1.bias": [64],
                "fc1.weight": [128, 1024],
                "fc1.bias": [128]
            }
        }
    },
    "summary": {
        "num_keys": 1,
        "contains_params_ema": false,
        "contains_params": true,
        "contains_optimizer": false
    }
}
```

### Explanation of Results

- **`file_path`**: The path to the analyzed `.pth` file.
- **`keys`**: The top-level keys present in the `.pth` file.
- **`details`**: Detailed analysis of each key, including:
  - **`state_dict`**: Describes model weights and their shapes if present.
  - **`optimizer`**: Indicates optimizer state keys if present.
  - **`params` or `params_ema`**: Describes parameter counts and shapes for these keys.
- **`summary`**: A high-level overview of the file contents.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributing

Contributions are welcome! If you have ideas for new features or improvements, feel free to submit a pull request.

---

## Support

TorchModelAnalyzer offers multiple ways for users to seek help or contribute feedback. Please use the appropriate channel for your query or request.

### Discussions

If you have questions, installation problems, or want to share feedback or ideas, please use the [Discussions](https://github.com/eoffermann/TorchModelAnalyzer/discussions) tab in the GitHub repository. 

- **Ask Questions**: Clarify usage or setup issues.
- **Feature Requests**: Share your ideas for new features. Requests may be moved to Issues for developer attention after being discussed.
- **Community Feedback**: Discuss your experiences or potential use cases with the community.

### Issues

Use the [Issues](https://github.com/eoffermann/TorchModelAnalyzer/issues) tab for bug reports or confirmed feature requests only. This helps keep the issue tracker organized and focused on actionable items for developers.

#### What to Include in a Bug Report

If you encounter a bug, please use the following template to submit a detailed issue:

1. **Title**: Write a concise title that summarizes the issue (e.g., "Error when loading .pth files with state_dict").
   
2. **Description**:
   - Clearly describe the problem you encountered.
   - Include the behavior you expected versus what actually happened.
   - Mention if this issue is consistent or intermittent.

3. **Steps to Reproduce**:
   - Provide a step-by-step guide to reproduce the issue. Example:
     ```plaintext
     1. Install dependencies using `pip install -r requirements.txt`.
     2. Run `python analyze.py /path/to/model.pth`.
     3. Observe the following error in the output: "KeyError: 'state_dict'."
     ```

4. **Environment Details**:
   - Python version (e.g., `3.10.11`).
   - Cuda version.
   - Pytorch version.
   - Operating System (e.g., `Windows 11`, `Ubuntu 20.04`).

5. **Logs and Error Messages**:
   - Include any error messages or stack traces you received.
   - Example:
     ```plaintext
     Traceback (most recent call last):
       File "analyze.py", line 45, in <module>
         data = torch.load(file_path, map_location="cpu")
     KeyError: 'state_dict'
     ```

6. **Additional Context**:
   - Share any other information you think is relevant (e.g., was the `.pth` file created with a custom model class?).

### Feature Request Workflow

If you have a feature request:
1. Start a **new discussion** in the [Discussions](https://github.com/eoffermann/TorchModelAnalyzer/discussions) tab under the "Ideas" category.
2. If there’s agreement or developer interest, the idea may be moved to an **Issue** for tracking.

---

By following these guidelines, you help ensure that the community and developers can address your concerns efficiently.

---

## Notes

- Ensure you have saved your files using `torch.save()` with compatible formats.
- This script works with PyTorch `.pth` files. It probably works just fine with `.pt` files and similar, as well, but hasn't been tested with them. Let us know in [Discussions](https://github.com/eoffermann/TorchModelAnalyzer/discussions) if you have success with other model formats.
- For models saved with `state_dict`, you will need the corresponding model definition to reload the model.
