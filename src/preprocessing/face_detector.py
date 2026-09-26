"""
Face Detector - Wrapper for face detection and alignment
Uses MTCNN from facenet-pytorch
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Union

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: facenet-pytorch not installed. Using OpenCV face detector fallback.")

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: insightface not installed. RetinaFace will not be available.")


class FaceDetector:
    """
    Face detector with support for MTCNN, RetinaFace (InsightFace), or OpenCV fallback.
    
    Features:
    - Face detection
    - Face cropping with margin
    - Multi-engine support
    """
    
    def __init__(
        self,
        device: str = 'cpu',
        min_face_size: int = 40,
        margin: int = 20,
        detector_type: str = 'mtcnn', # 'mtcnn', 'retinaface', or 'opencv'
    ):
        """
        Args:
            device: Device for detectors ('cpu' or 'cuda')
            min_face_size: Minimum face size to detect
            margin: Margin around detected face
            detector_type: Type of detector to use
        """
        self.margin = margin
        self.device = device
        self.min_face_size = min_face_size
        self.detector_type = detector_type.lower()
        
        # Initialize the requested detector
        if self.detector_type == 'retinaface' and INSIGHTFACE_AVAILABLE:
            self._init_retinaface()
        elif self.detector_type == 'mtcnn' and MTCNN_AVAILABLE:
            self._init_mtcnn()
        else:
            if detector_type != 'opencv':
                print(f"Warning: {detector_type} not available. Falling back to OpenCV.")
            self._init_opencv()

    def _init_mtcnn(self):
        """Initialize MTCNN detector."""
        self.detector = MTCNN(
            image_size=224,
            margin=self.margin,
            min_face_size=self.min_face_size,
            device=self.device,
            post_process=False,
        )
        self.detector_engine = 'mtcnn'
        print("Using MTCNN face detector")

    def _init_retinaface(self):
        """Initialize InsightFace/RetinaFace detector Optimized for CPU."""
        try:
            # We only need the 'detection' module. 
            # Skipping 'landmark_2d_106', 'landmark_3d_68', 'genderage', 'recognition' to save CPU/Memory.
            self.detector = FaceAnalysis(
                name='buffalo_s', 
                providers=['CPUExecutionProvider'],
                allowed_modules=['detection']
            )
            self.detector.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(320, 320))
            self.detector_engine = 'retinaface'
            print("Using Optimized InsightFace (RetinaFace - Detection Only)")
        except Exception as e:
            print(f"Error initializing InsightFace: {e}. Falling back to OpenCV.")
            self._init_opencv()

    def _init_opencv(self):
        """Initialize OpenCV Haar Cascade fallback."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.detector_engine = 'opencv'
        print("Using OpenCV Haar Cascade face detector")
    
    def detect_faces(
        self, 
        image: Union[np.ndarray, Image.Image],
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in image.
        
        Args:
            image: Input image (BGR numpy array or PIL Image)
        
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB for processing
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Apply backlighting correction to reveal dark silhouetted faces
            rgb_image = correct_backlighting(rgb_image)
            pil_image = Image.fromarray(rgb_image)
        else:
            pil_image = image
            rgb_image = np.array(image)
            rgb_image = correct_backlighting(rgb_image)
            pil_image = Image.fromarray(rgb_image)
        
        if self.detector_engine == 'mtcnn':
            boxes, probs = self.detector.detect(pil_image)
            
            if boxes is None:
                return []
            
            # Convert to (x, y, w, h) format
            faces = []
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2 - x1, y2 - y1))
            return faces
            
        elif self.detector_engine == 'retinaface':
            # InsightFace works with RGB numpy arrays
            faces_info = self.detector.get(rgb_image)
            
            faces = []
            for face in faces_info:
                x1, y1, x2, y2 = face.bbox.astype(int)
                # Keep detections larger than min_face_size
                if (x2 - x1) >= self.min_face_size and (y2 - y1) >= self.min_face_size:
                    faces.append((x1, y1, x2 - x1, y2 - y1))
            return faces
            
        else:
            # OpenCV detection
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            faces = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size)
            )
            return [tuple(face) for face in faces]
    
    def crop_face(
        self,
        image: Union[np.ndarray, Image.Image],
        bbox: Tuple[int, int, int, int],
        output_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Crop and resize face from image.
        
        Args:
            image: Input image
            bbox: Bounding box (x, y, w, h)
            output_size: Output size (width, height)
        
        Returns:
            Cropped face image
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        x, y, w, h = bbox
        height, width = image.shape[:2]
        
        # Add margin
        x1 = max(0, x - self.margin)
        y1 = max(0, y - self.margin)
        x2 = min(width, x + w + self.margin)
        y2 = min(height, y + h + self.margin)
        
        # Crop
        face = image[y1:y2, x1:x2]
        
        # Resize
        face = cv2.resize(face, output_size)
        
        return face
    
    def detect_and_crop(
        self,
        image: Union[np.ndarray, Image.Image],
        output_size: Tuple[int, int] = (224, 224),
    ) -> List[np.ndarray]:
        """
        Detect faces and crop them.
        
        Args:
            image: Input image
            output_size: Output size for cropped faces
        
        Returns:
            List of cropped face images
        """
        if isinstance(image, Image.Image):
            np_image = np.array(image)
        else:
            np_image = image.copy()
            # Convert BGR to RGB for processing
            np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        
        faces = self.detect_faces(np_image)
        cropped_faces = []
        
        for bbox in faces:
            face = self.crop_face(np_image, bbox, output_size)
            cropped_faces.append(face)
        
        return cropped_faces
    
    def detect_largest_face(
        self,
        image: Union[np.ndarray, Image.Image],
        output_size: Tuple[int, int] = (224, 224),
    ) -> Optional[np.ndarray]:
        """
        Detect and crop the largest face.
        
        Args:
            image: Input image
            output_size: Output size
        
        Returns:
            Cropped face or None if no face detected
        """
        if isinstance(image, Image.Image):
            np_image = np.array(image)
        else:
            np_image = image
        
        faces = self.detect_faces(np_image)
        
        if not faces:
            return None
        
        # Find largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        
        return self.crop_face(np_image, largest_face, output_size)


def enhance_face_illumination(face_img: np.ndarray) -> np.ndarray:
    """
    Adaptive illumination enhancement for cropped face images.
    If the image is under-exposed/low-light (mean brightness < 90),
    applies a combination of gamma correction and CLAHE to restore details.
    
    Args:
        face_img: Crop of face (BGR numpy array)
        
    Returns:
        Enhanced BGR numpy array
    """
    if face_img is None or face_img.size == 0:
        return face_img
        
    # Calculate average brightness in grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 90:
        # 1. Apply adaptive gamma correction if extremely dark
        if mean_brightness < 50:
            # Gamma < 1.0 brightens the image
            gamma = 0.65
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            face_img = cv2.LUT(face_img, table)
            
        # 2. Apply CLAHE on the Lightness channel in LAB space
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        face_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
    return face_img


def correct_backlighting(image_rgb: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE on the Lightness channel of the image to correct backlighting / silhouetting.
    This makes dark faces in front of bright backgrounds visible to the face detector.
    
    Args:
        image_rgb: Input RGB image (numpy array)
        
    Returns:
        Enhanced RGB image (numpy array)
    """
    if image_rgb is None or image_rgb.size == 0:
        return image_rgb
        
    # Convert RGB to LAB color space
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to equalize lightness distribution
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge and convert back to RGB
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb


if __name__ == "__main__":
    # Test face detector
    detector = FaceDetector(device='cpu')
    print("Face detector loaded successfully")
