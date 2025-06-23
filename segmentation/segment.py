import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
import torchvision.transforms as T
from PIL import Image
from segmentation.visualization import visualize_instance, visualize_semantic

# === DEVICE SETUP ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === SEGMENTATION MODEL WRAPPER ===
class SegmentationModels:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instance Segmentation: Mask R-CNN with updated weights
        instance_weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        self.instance_model = maskrcnn_resnet50_fpn(weights=instance_weights)
        self.instance_model.to(self.device)
        self.instance_model.eval()

        # Semantic Segmentation: DeepLabV3 with updated weights
        semantic_weights = DeepLabV3_ResNet101_Weights.DEFAULT
        self.semantic_model = deeplabv3_resnet101(weights=semantic_weights)
        self.semantic_model.to(self.device)
        self.semantic_model.eval()

    def predict_instance(self, image_tensor):
        with torch.no_grad():
            output = self.instance_model(image_tensor.to(self.device))[0]
        return output

    def predict_semantic(self, image_tensor):
        with torch.no_grad():
            output = self.semantic_model(image_tensor.to(self.device))['out']
            mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
        return mask

# === MAIN FUNCTION TO SEGMENT AN IMAGE ===
def segment_image(pil_img):
    # Load models
    model = SegmentationModels(device=DEVICE)

    # Resize image to match model input size (fix shape mismatch in visualization)
    resized_pil_img = pil_img.resize((384, 384))

    # Preprocess image
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(resized_pil_img).unsqueeze(0).to(DEVICE)

    # Instance Segmentation
    instance_output = model.predict_instance(img_tensor)
    masks = instance_output['masks'] > 0.5
    boxes = instance_output['boxes']
    labels = instance_output['labels']

    # Semantic Segmentation
    semantic_mask = model.predict_semantic(img_tensor)

    return resized_pil_img, masks, boxes, labels, semantic_mask

# === EXAMPLE USAGE ===
if __name__ == "__main__":
    image_path = r"D:\Coding file\Zidio_development\image_captioning_segmentation\segmentation\image.png"
    image = Image.open(image_path).convert("RGB")

    resized_img, masks, boxes, labels, semantic_mask = segment_image(image)

    print("Instance Segmentation Output")
    print(f"- Detected {len(masks)} objects")
    print(f"- Boxes shape: {boxes.shape}")
    print(f"- Labels: {labels.tolist()}")

    print("\nSemantic Segmentation Output")
    print(f"- Semantic mask shape: {semantic_mask.shape}")

    # Visualization
    visualize_instance(resized_img, masks, boxes, labels)
    visualize_semantic(resized_img, semantic_mask)