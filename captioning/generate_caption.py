# # captioning/generate_caption.py

# import torch
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
# from model import get_model  # Assumes get_model() returns (model, tokenizer)

# # Set computation device
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def extract_features(img_path: str) -> torch.Tensor:
#     """
#     Extract image features using a pretrained ResNet50 (excluding final FC layer).
    
#     Args:
#         img_path (str): Path to the input image.
    
#     Returns:
#         torch.Tensor: Extracted feature vector of shape [2048].
#     """
#     # Load pretrained ResNet50 and remove classification layer
#     resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
#     resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
#     resnet.eval().to(DEVICE)

#     # Image preprocessing pipeline
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406], 
#             std=[0.229, 0.224, 0.225]
#         )
#     ])

#     # Load and preprocess image
#     image = Image.open(img_path).convert("RGB")
#     image_tensor = transform(image).unsqueeze(0).to(DEVICE)  # Shape: [1, 3, 224, 224]

#     # Extract features
#     with torch.no_grad():
#         features = resnet(image_tensor).squeeze()  # Shape: [2048]
    
#     return features


# def generate_caption(image_path: str) -> str:
#     """
#     Generate a caption for the given image using a custom image captioning model.
    
#     Args:
#         image_path (str): Path to the image.
    
#     Returns:
#         str: Generated image caption.
#     """
#     # Load model and tokenizer
#     model, tokenizer = get_model()
#     model.eval().to(DEVICE)

#     # Extract image features
#     image_feat = extract_features(image_path).unsqueeze(0).to(DEVICE)  # Shape: [1, 2048]

#     # Initialize input with CLS token
#     generated_ids = torch.tensor([[tokenizer.cls_token_id]], device=DEVICE)

#     # Generate caption token-by-token
#     for _ in range(32):  # Max caption length
#         logits = model(image_feat, generated_ids)  # Forward pass
#         next_token_logits = logits[:, -1, :]       # Get logits for last token
#         next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # Greedy decoding
#         generated_ids = torch.cat([generated_ids, next_token_id], dim=1)  # Append next token

#         # Stop if SEP token is generated
#         if next_token_id.item() == tokenizer.sep_token_id:
#             break

#     # Decode token IDs into a human-readable sentence
#     caption = tokenizer.decode(generated_ids.squeeze().tolist(), skip_special_tokens=True)
#     return caption


# # Optional: Run script standalone for testing
# if __name__ == "__main__":
#     test_image_path = r"D:\Coding file\Zidio_development\image_captioning_segmentation\captioning\img1.jpg"
#     caption = generate_caption(test_image_path)
#     print("🖼️ Generated Caption:", caption)


import torch
from PIL import Image
from captioning.model import get_model  # Assumes get_model() returns (model, processor/tokenizer)

# Automatically select GPU if available, else fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_caption(image: Image.Image) -> str:
    """
    Generate a descriptive caption for the given input image.

    Args:
        image (PIL.Image.Image): Input image to caption.

    Returns:
        str: Generated image caption.
    """

    # Load the image captioning model and its corresponding processor/tokenizer
    model, processor = get_model()
    model.to(DEVICE)
    model.eval()

    # Preprocess the image to match model input requirements
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    # Generate caption using greedy decoding (can be replaced with beam search if needed)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=30)

    # Decode the token IDs to human-readable text
    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption


# Optional standalone test
if __name__ == "__main__":
    # Provide path to a test image
    image_path = r"D:\Coding file\Zidio_development\image_captioning_segmentation\captioning\img1.jpg"
    
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    
    # Generate and print caption
    caption = generate_caption(image)
    print("🖼️ Caption:", caption)


