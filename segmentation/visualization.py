# segmentation/visualize.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image, to_tensor
import random
import cv2


def random_color() -> list:
    """
    Generate a random RGB color.

    Returns:
        list: A list of 3 integers representing an RGB color.
    """
    return [random.randint(0, 255) for _ in range(3)]


def visualize_instance(image, masks, boxes, labels, label_map=None):
    """
    Visualize instance segmentation by drawing colored masks and bounding boxes.

    Args:
        image (PIL.Image): Original input image.
        masks (Tensor): Predicted binary masks (shape: [N, 1, H, W]).
        boxes (Tensor): Bounding boxes (shape: [N, 4]).
        labels (Tensor): Class labels corresponding to each instance.
        label_map (dict, optional): Mapping of label IDs to class names.

    Returns:
        None. Displays the annotated image using matplotlib.
    """
    # Convert image to tensor and scale to 0–255
    image_tensor = to_tensor(image).mul(255).byte().cpu()

    # Move all tensors to CPU
    masks = masks.cpu()
    boxes = boxes.cpu()
    labels = labels.cpu()

    # Draw masks using different colors
    mask_img = image_tensor.clone()
    for i in range(min(len(masks), 10)):  # Limit to 10 masks for visibility
        mask = masks[i][0]  # Single channel binary mask
        color = torch.tensor(random_color(), dtype=torch.uint8).view(3, 1, 1)
        mask_img = torch.where(mask.bool(), color, mask_img)

    # Convert label IDs to strings
    if label_map:
        label_names = [label_map.get(l.item(), str(l.item())) for l in labels]
    else:
        label_names = [str(l.item()) for l in labels]

    # Draw bounding boxes with labels
    if len(boxes) > 0:
        image_with_boxes = draw_bounding_boxes(
            mask_img, boxes=boxes, labels=label_names, colors="red", width=2
        )
    else:
        image_with_boxes = mask_img

    # Convert to PIL and display
    result_img = to_pil_image(image_with_boxes)
    plt.figure(figsize=(8, 8))
    plt.imshow(result_img)
    plt.title("Instance Segmentation")
    plt.axis("off")
    plt.show()


def visualize_semantic(image, semantic_mask, num_classes=21):
    """
    Overlay a semantic segmentation mask on the input image.

    Args:
        image (PIL.Image): Original input image.
        semantic_mask (ndarray): 2D array of class IDs with shape [H, W].
        num_classes (int): Total number of semantic classes (used for colormap).

    Returns:
        None. Displays the overlay using matplotlib.
    """
    # Generate a colormap for each class
    colormap = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

    # Apply colormap to semantic mask (convert class IDs to RGB)
    semantic_color = colormap[semantic_mask]

    # Resize original image to match the mask shape
    image_np = np.array(image.resize(semantic_mask.shape[::-1]))  # reverse HxW -> WxH

    # Blend image and mask using OpenCV
    overlay = cv2.addWeighted(image_np, 0.5, semantic_color, 0.5, 0)

    # Display result
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.title("Semantic Segmentation")
    plt.axis("off")
    plt.show()
