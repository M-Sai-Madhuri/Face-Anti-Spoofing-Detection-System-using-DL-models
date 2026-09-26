"""
Face Encoder - Extract face identity embeddings using InceptionResnetV1.
Uses facenet-pytorch (already in requirements) pretrained on VGGFace2.
Produces a 512-dimensional L2-normalized embedding per face.
Supports both single and batched inference.
"""

import logging
import cv2
import numpy as np
import torch
from typing import List
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

from src.preprocessing import enhance_face_illumination


class FaceEncoder:
    """
    Extracts compact face identity embeddings for recognition.
    
    Uses InceptionResnetV1 from facenet-pytorch pretrained on VGGFace2.
    Input: cropped face image (BGR numpy array, any size)
    Output: 512-d L2-normalized embedding
    """

    def __init__(self, device: str = 'auto'):
        """
        Args:
            device: 'auto', 'cuda', or 'cpu'
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load InceptionResnetV1 pretrained on VGGFace2
        from facenet_pytorch import InceptionResnetV1
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Standard transforms for InceptionResnetV1
        # Expects 160x160 RGB images normalized to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        print(f"[FaceEncoder] Loaded InceptionResnetV1 (VGGFace2) on {self.device}")

    @torch.no_grad()
    def encode(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract a 512-d identity embedding from a cropped face image.

        Args:
            face_img: BGR numpy array of a cropped face (any size)

        Returns:
            512-d L2-normalized numpy embedding
        """
        # Apply low-light enhancement adaptively
        face_img = enhance_face_illumination(face_img)

        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)

        # Preprocess
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        # Forward pass
        embedding = self.model(tensor)

        # L2 normalize
        embedding = embedding / embedding.norm(dim=1, keepdim=True)

        return embedding.cpu().numpy().flatten()

    def preprocess(self, face_img: np.ndarray) -> torch.Tensor:
        """
        Preprocess a face image to a tensor without running inference.

        Args:
            face_img: BGR numpy array of a cropped face (any size)

        Returns:
            Preprocessed tensor of shape (3, 160, 160)
        """
        # Apply low-light enhancement adaptively
        face_img = enhance_face_illumination(face_img)

        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(face_rgb)
        return self.transform(pil_image)

    @torch.no_grad()
    def encode_batch(self, face_imgs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Extract 512-d identity embeddings from multiple face images in a
        single forward pass (batched inference).

        Performance: ~5-10x faster than sequential encode() for 15+ images,
        because the GPU/CPU overhead of model invocation is amortized.

        Args:
            face_imgs: List of BGR numpy arrays (cropped faces, any size)

        Returns:
            List of 512-d L2-normalized numpy embeddings
        """
        if len(face_imgs) == 0:
            return []

        # Step 1: Preprocess all images to tensors
        tensors = []
        for img in face_imgs:
            tensors.append(self.preprocess(img))

        # Step 2: Stack into batch tensor: [N, 3, 160, 160]
        batch = torch.stack(tensors).to(self.device)

        # Step 3: Single forward pass
        embeddings = self.model(batch)  # [N, 512]

        # Step 4: L2 normalize each embedding
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

        # Step 5: Convert to list of numpy arrays
        embeddings_np = embeddings.cpu().numpy()
        result = [embeddings_np[i].flatten() for i in range(embeddings_np.shape[0])]

        logger.debug(f"Batch encoded {len(result)} faces in single forward pass")
        return result

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two L2-normalized embeddings.

        Args:
            emb1: 512-d embedding
            emb2: 512-d embedding

        Returns:
            Cosine similarity score in [-1, 1] (higher = more similar)
        """
        return float(np.dot(emb1, emb2))


if __name__ == "__main__":
    encoder = FaceEncoder(device='cpu')
    # Test with a dummy image
    dummy = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    emb = encoder.encode(dummy)
    print(f"Embedding shape: {emb.shape}")
    print(f"Embedding norm: {np.linalg.norm(emb):.4f}")
    sim = FaceEncoder.compute_similarity(emb, emb)
    print(f"Self-similarity: {sim:.4f}")
