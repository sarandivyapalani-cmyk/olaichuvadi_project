import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.ndimage import label, find_objects

class TamilPalmLeafSegmenter:
    """
    Module 2: Text Line and Character Segmentation
    - Line detection using projection profiles
    - Character segmentation using connected components
    """
    
    def __init__(self):
        self.min_line_height = 20
        self.min_char_width = 10
        
    def segment_lines(self, enhanced_image):
        """
        Segment palm leaf image into individual text lines
        """
        # Binarize
        _, binary = cv2.threshold(
            enhanced_image, 
            0, 255, 
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # Horizontal projection profile
        h_projection = np.sum(binary == 0, axis=1)
        
        # Find line boundaries
        line_boundaries = []
        in_line = False
        start = 0
        
        threshold = np.mean(h_projection) * 0.3
        
        for i, value in enumerate(h_projection):
            if value > threshold and not in_line:
                in_line = True
                start = i
            elif value <= threshold and in_line:
                in_line = False
                end = i
                if end - start > self.min_line_height:
                    line_boundaries.append((start, end))
        
        # Extract line images
        lines = []
        line_images = []
        
        for i, (start, end) in enumerate(line_boundaries):
            # Add padding
            start = max(0, start - 5)
            end = min(binary.shape[0], end + 5)
            
            line_img = binary[start:end, :]
            line_img = self.clean_line_image(line_img)
            
            if line_img.size > 0:
                lines.append((start, end))
                line_images.append(line_img)
        
        return lines, line_images
    
    def segment_characters(self, line_image):
        """
        Segment line image into individual characters
        """
        # Vertical projection for character segmentation
        v_projection = np.sum(line_image == 0, axis=0)
        
        # Find character boundaries
        char_boundaries = []
        in_char = False
        start = 0
        
        threshold = np.mean(v_projection) * 0.2
        
        for i, value in enumerate(v_projection):
            if value > threshold and not in_char:
                in_char = True
                start = i
            elif value <= threshold and in_char:
                in_char = False
                end = i
                if end - start > self.min_char_width:
                    char_boundaries.append((start, end))
        
        # Extract character images
        characters = []
        for start, end in char_boundaries:
            char_img = line_image[:, start:end]
            char_img = self.normalize_character(char_img)
            characters.append(char_img)
        
        return characters
    
    def clean_line_image(self, line_image):
        """
        Clean line image - remove noise, deskew
        """
        # Morphological cleaning
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(line_image, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Deskew
        coords = np.column_stack(np.where(cleaned == 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.5:
                (h, w) = cleaned.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                cleaned = cv2.warpAffine(
                    cleaned, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
        
        return cleaned
    
    def normalize_character(self, char_img):
        """
        Normalize character image for recognition
        """
        # Resize to standard size
        target_size = (64, 64)
        resized = cv2.resize(char_img, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values
        normalized = resized / 255.0
        
        return normalized
