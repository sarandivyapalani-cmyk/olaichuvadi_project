import matplotlib.pyplot as plt
import cv2
import numpy as np

def display_comparison(original, enhanced, lines=None):
    """
    Display before/after comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Manuscript')
    axes[0].axis('off')
    
    axes[1].imshow(enhanced, cmap='gray')
    axes[1].set_title('Enhanced Manuscript')
    axes[1].axis('off')
    
    if lines:
        for start, end in lines[:3]:
            axes[1].axhline(y=start, color='red', linestyle='--', alpha=0.5)
            axes[1].axhline(y=end, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig

def plot_confidence_distribution(confidences):
    """
    Plot confidence scores
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(confidences)), confidences)
    ax.set_xlabel('Character Position')
    ax.set_ylabel('Confidence Score')
    ax.set_title('Recognition Confidence per Character')
    ax.set_ylim(0, 1)
    return fig
