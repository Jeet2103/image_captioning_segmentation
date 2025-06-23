# app/utils.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def display_masks(image, masks, boxes, labels, label_map=None):
    """
    Display instance segmentation results on a given image.
    
    Args:
        image (PIL.Image): The original input image.
        masks (Tensor): A tensor of predicted binary masks for each detected instance.
        boxes (Tensor): A tensor of bounding box coordinates (x1, y1, x2, y2).
        labels (Tensor): A tensor of class labels for each instance.
        label_map (dict, optional): A dictionary mapping label IDs to human-readable class names.

    Returns:
        None. Displays the annotated image using Streamlit.
    """

    # Convert the input image to a NumPy array for visualization
    np_image = np.array(image)

    # Create a matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np_image)
    ax.axis("off")

    # Limit the number of instances shown (to avoid clutter)
    max_instances = min(len(masks), 20)

    for i in range(max_instances):
        # Extract mask and convert to NumPy array
        mask = masks[i][0].cpu().numpy()

        # Get bounding box coordinates
        x1, y1, x2, y2 = boxes[i].cpu().numpy()

        # Retrieve label and its corresponding class name
        label = labels[i].item()
        class_name = label_map[label] if label_map and label in label_map else f"Class {label}"

        # Draw the mask contour in red
        ax.contour(mask, colors='red', linewidths=1)

        # Draw the bounding box in lime green
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, color='lime', linewidth=1.5
        )
        ax.add_patch(rect)

        # Add the class label as a caption above the bounding box
        ax.text(
            x1, y1 - 5, class_name,
            color='yellow', fontsize=8,
            backgroundcolor='black'
        )

    # Display the final figure using Streamlit
    st.pyplot(fig)
