import numpy as np
import tensorflow as tf
from transformers import AutoImageProcessor, AutoModelForImageClassification
import cv2

class AncientTamilRecognizer:
    """
    Module 3: Ancient Tamil Character Recognition
    - Transfer learning from pretrained models
    - Supports Vattezhuthu and Grantha scripts
    """
    
    def __init__(self):
        # Dictionary mapping ancient to modern Tamil
        self.ancient_to_modern = {
            # Vattezhuthu to Modern Tamil mapping
            '𑀅': 'அ',
            '𑀆': 'ஆ',
            '𑀇': 'இ',
            '𑀈': 'ஈ',
            '𑀉': 'உ',
            '𑀊': 'ஊ',
            '𑀓': 'க',
            '𑀕': 'க',
            '𑀗': 'ங',
            '𑀘': 'ச',
            '𑀚': 'ச',
            '𑀜': 'ஞ',
            '𑀝': 'ட',
            '𑀞': 'ட',
            '𑀡': 'ண',
            '𑀢': 'த',
            '𑀤': 'த',
            '𑀦': 'ந',
            '𑀧': 'ப',
            '𑀨': 'ப',
            '𑀫': 'ம',
            '𑀬': 'ய',
            '𑀭': 'ர',
            '𑀮': 'ல',
            '𑀯': 'வ',
            '𑀰': 'ஷ',
            '𑀲': 'ச',
            '𑀳': 'ஹ'
        }
        
        # Load pretrained model (you'll fine-tune this)
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                "google/vit-base-patch16-224"
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                "google/vit-base-patch16-224",
                num_labels=100,  # Number of Tamil characters
                ignore_mismatched_sizes=True
            )
        except:
            print("Using fallback recognition method")
            self.model = None
    
    def recognize(self, line_images, script_type="Vattezhuthu"):
        """
        Recognize ancient Tamil characters from line images
        """
        recognized_text = ""
        confidence_scores = []
        char_details = []
        
        for line_img in line_images:
            # Segment line into characters
            characters = self.segment_characters_fallback(line_img)
            
            for char_img in characters:
                # Recognize character
                modern_char, confidence = self.recognize_character(char_img)
                
                if modern_char:
                    recognized_text += modern_char
                    confidence_scores.append(confidence)
                    char_details.append(confidence)
        
        return recognized_text, confidence_scores, char_details
    
    def recognize_character(self, char_img):
        """
        Recognize individual character
        """
        if self.model is not None:
            # Use transformer model
            inputs = self.processor(char_img, return_tensors="pt")
            outputs = self.model(**inputs)
            probs = tf.nn.softmax(outputs.logits, axis=-1)
            confidence = float(np.max(probs))
            pred_class = int(np.argmax(probs))
            
            # Map to Tamil character (simplified - you'll have real mapping)
            modern_char = list(self.ancient_to_modern.values())[pred_class % len(self.ancient_to_modern)]
            
            return modern_char, confidence
        else:
            # Fallback: rule-based recognition
            return self.rule_based_recognition(char_img)
    
    def rule_based_recognition(self, char_img):
        """
        Fallback recognition using contour matching
        """
        # Simplified - in production, use actual trained model
        h, w = char_img.shape if len(char_img.shape) == 2 else char_img.shape[:2]
        
        # Basic shape-based rules
        aspect_ratio = w / h
        black_pixels = np.sum(char_img < 50)
        white_pixels = np.sum(char_img > 200)
        
        # Very simplified mapping (you'll implement proper matching)
        if aspect_ratio < 0.5:
            return 'க', 0.75
        elif aspect_ratio > 2.0:
            return 'அ', 0.70
        elif black_pixels < 500:
            return 'இ', 0.65
        else:
            return 'த', 0.60
    
    def segment_characters_fallback(self, line_img):
        """
        Character segmentation fallback
        """
        # Simple connected component analysis
        if len(line_img.shape) == 3:
            gray = cv2.cvtColor(line_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = line_img
            
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        characters = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 10 and h > 20:  # Filter noise
                char = binary[y:y+h, x:x+w]
                char = cv2.resize(char, (64, 64))
                characters.append(char)
        
        return characters
