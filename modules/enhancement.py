import cv2
import numpy as np
from skimage import restoration, exposure
from skimage.filters import unsharp_mask
import tensorflow as tf
from scipy.ndimage import median_filter

class PalmLeafEnhancer:
    """
    Module 1: Olaichuvadi Image Enhancement
    - Background removal
    - Contrast enhancement
    - Inpainting for damaged regions
    """
    
    def __init__(self):
        self.denoise_strength = 10
        self.contrast_clip = 2.0
        
    def enhance(self, image, denoise_strength=10, contrast_clip=2.0):
        """
        Complete enhancement pipeline
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Step 1: Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Step 2: Denoising (Non-local means)
        denoised = cv2.fastNlMeansDenoising(
            enhanced, 
            None, 
            denoise_strength, 
            7, 21
        )
        
        # Step 3: Unsharp masking for edge enhancement
        sharpened = unsharp_mask(denoised, radius=1.0, amount=1.5)
        sharpened = (sharpened * 255).astype(np.uint8)
        
        # Step 4: Background normalization
        # Remove palm leaf texture using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        background = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
        normalized = cv2.divide(sharpened, background, scale=255)
        
        # Step 5: Binary threshold for damaged regions
        _, binary = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Step 6: Simple inpainting for small damages
        if np.sum(binary == 0) < 0.3 * binary.size:  # Less than 30% damage
            inpainted = cv2.inpaint(
                normalized.astype(np.uint8),
                (binary == 0).astype(np.uint8),
                3,
                cv2.INPAINT_TELEA
            )
        else:
            inpainted = normalized
        
        # Step 7: Final contrast adjustment
        final = exposure.rescale_intensity(inpainted, out_range=(0, 255))
        final = final.astype(np.uint8)
        
        return final
    
    def remove_background(self, image):
        """
        Remove palm leaf background texture
        """
        # Gaussian blur for texture removal
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Background subtraction
        background_removed = cv2.subtract(image, blurred)
        
        # Normalize
        background_removed = cv2.normalize(
            background_removed, 
            None, 
            0, 255, 
            cv2.NORM_MINMAX
        )
        
        return background_removed
    
    def inpaint_damaged(self, image, damage_mask=None):
        """
        Inpaint damaged/cracked regions
        """
        if damage_mask is None:
            # Detect potential damage (cracks, holes)
            _, damage_mask = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)
            damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_OPEN, np.ones((3,3)))
        
        # Inpainting
        inpainted = cv2.inpaint(
            image.astype(np.uint8),
            damage_mask.astype(np.uint8),
            5,
            cv2.INPAINT_TELEA
        )
        
        return inpainted
