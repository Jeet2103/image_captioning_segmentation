# create_project_structure.py

import os
import pathlib

def create_project_structure(base_path="../image_captioning_segmentation"):
    """
    Creates the directory structure and essential boilerplate files
    for the Image Captioning and Segmentation project.
    
    Args:
        base_path (str): Base directory path for the project structure.
    """

    # ========================
    # Define project structure
    # ========================
    project_structure = {
        base_path: {
            "app": [
                "__init__.py",
                "main.py",
                "utils.py"
            ],
            "captioning": [
                "__init__.py",
                "model.py",
                "generate_caption.py",
                "train_captioning.py"
            ],
            "segmentation": [
                "__init__.py",
                "model.py",
                "segment.py",
                "train_segmentation.py"
            ],
            "data": {
                "captions": [],
                "images": []
            },
            "notebooks": [
                "training_notebook.ipynb"
            ],
            "evaluations": [],
            "requirements.txt": None,
            "README.md": None,
            "config.py": None
        }
    }

    def create_structure(current_path, structure):
        """
        Recursively creates the specified project folder and file hierarchy.
        
        Args:
            current_path (str): Current directory path.
            structure (dict | list | None): Folder or file structure.
        """
        for name, content in structure.items():
            path = os.path.join(current_path, name)

            if isinstance(content, dict):
                os.makedirs(path, exist_ok=True)
                print(f"📁 Created directory: {path}")
                create_structure(path, content)

            elif isinstance(content, list):
                os.makedirs(path, exist_ok=True)
                print(f"📁 Created directory: {path}")
                for file_name in content:
                    file_path = os.path.join(path, file_name)
                    pathlib.Path(file_path).touch()
                    print(f"📄 Created file: {file_path}")

            elif content is None:
                pathlib.Path(path).touch()
                print(f"📄 Created file: {path}")

    # ========================
    # Initialize base directory
    # ========================
    os.makedirs(base_path, exist_ok=True)
    print(f"📦 Created base directory: {base_path}")

    # Create folder and file structure
    create_structure(base_path, project_structure[base_path])

    # ========================
    # Write basic template files
    # ========================

    # README.md
    with open(os.path.join(base_path, "README.md"), "w") as f:
        f.write("# 🖼️ Image Captioning and Segmentation Project\n\n"
                "This project combines image captioning and segmentation using deep learning models "
                "including BLIP, ResNet, and Mask R-CNN.\n")

    # requirements.txt
    with open(os.path.join(base_path, "requirements.txt"), "w") as f:
        f.write(
            "# Required Python packages\n"
            "torch\n"
            "torchvision\n"
            "transformers\n"
            "streamlit\n"
            "numpy\n"
            "pillow\n"
            "matplotlib\n"
            "jupyter\n"
        )

    # config.py
    with open(os.path.join(base_path, "config.py"), "w") as f:
        f.write(
            '"""Project configuration settings."""\n\n'
            'import torch\n\n'
            'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"\n'
            'DATA_PATH = "data"\n'
            'MODEL_PATH = "models"\n'
        )


if __name__ == "__main__":
    create_project_structure()
