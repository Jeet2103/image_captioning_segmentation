# captioning/model.py

# import torch
# import torch.nn as nn
# from transformers import BlipProcessor

# # Set device to GPU if available, else fallback to CPU
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class CaptioningModel(nn.Module):
#     """
#     A custom LSTM-based image captioning model.
#     Projects 2048-dim image features into a hidden space and decodes a caption using LSTM.
#     """

#     def __init__(self, feature_dim: int, hidden_dim: int, vocab_size: int, max_length: int = 32):
#         """
#         Args:
#             feature_dim (int): Dimension of input image features (e.g., 2048 from ResNet50).
#             hidden_dim (int): Hidden size for the LSTM network.
#             vocab_size (int): Vocabulary size from the tokenizer.
#             max_length (int): Maximum length of the generated captions.
#         """
#         super().__init__()

#         # Project 2048-dim image features to hidden_dim
#         self.img_proj = nn.Linear(feature_dim, hidden_dim)

#         # Token embedding layer
#         self.embedding = nn.Embedding(vocab_size, hidden_dim)

#         # LSTM decoder
#         self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

#         # Output layer to predict next token
#         self.fc = nn.Linear(hidden_dim, vocab_size)

#         self.max_length = max_length

#     def forward(self, image_feat: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through the model.

#         Args:
#             image_feat (Tensor): Extracted image features, shape [B, 2048].
#             input_ids (Tensor): Token IDs for input sequence, shape [B, L].

#         Returns:
#             Tensor: Logits over vocabulary for each time step, shape [B, L, vocab_size].
#         """
#         batch_size = image_feat.size(0)

#         # Project image features and prepend to token embeddings
#         img_emb = self.img_proj(image_feat).unsqueeze(1)      # Shape: [B, 1, H]
#         text_emb = self.embedding(input_ids[:, :-1])          # Shape: [B, L-1, H]
#         seq_input = torch.cat([img_emb, text_emb], dim=1)     # Shape: [B, L, H]

#         # Decode with LSTM
#         out, _ = self.lstm(seq_input)

#         # Project LSTM outputs to vocabulary logits
#         logits = self.fc(out)
#         return logits


# def get_model():
#     """
#     Loads the custom image captioning model and tokenizer (based on BLIP tokenizer).

#     Returns:
#         tuple:
#             - model (CaptioningModel): Trained PyTorch model for caption generation.
#             - tokenizer (PreTrainedTokenizer): Tokenizer for text processing.
#     """
#     # Load tokenizer from BLIP's processor (for compatibility with BLIP-trained vocab)
#     processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
#     tokenizer = processor.tokenizer
#     vocab_size = tokenizer.vocab_size

#     # Initialize model
#     model = CaptioningModel(
#         feature_dim=2048,
#         hidden_dim=512,
#         vocab_size=vocab_size
#     )

#     # Load pretrained weights
#     model_path = "captioning/feature_model/model.pt"
#     model.load_state_dict(torch.load(model_path, map_location=DEVICE))

#     # Move to device and set to eval mode
#     model = model.to(DEVICE)
#     model.eval()

#     return model, tokenizer


import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set device to GPU if available, else fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model():
    """
    Load the BLIP (Bootstrapped Language Image Pretraining) model and its processor
    for image captioning tasks.

    Returns:
        tuple:
            - model (BlipForConditionalGeneration): Pretrained BLIP captioning model.
            - processor (BlipProcessor): Corresponding processor for preprocessing input images.
    """
    # Load the image processor (handles image preprocessing and tokenization)
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_fast=True
    )

    # Load the pretrained BLIP model for conditional generation
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE)
    
    model.eval()  # Set model to evaluation mode
    
    return model, processor


