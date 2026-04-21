import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_image(image_path):
    """Load retinal fundus image and return BGR + RGB versions."""
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb

def get_disk_kernel(radius):
    """Create a circular (disk-shaped) structuring element."""
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return kernel

def phase1_preprocessing(img_rgb, kernel_radius=7):
    """
    Phase 1: Green channel extraction + morphological light reflex removal.
    
    T_output = Bottom_hat(green) - Top_hat(green)
    
    Returns the processed image and intermediate steps for visualization.
    """
    # Step 1: Extract green channel
    green = img_rgb[:, :, 1]
    
    # Step 2: Build structuring element
    kernel = get_disk_kernel(kernel_radius)
    
    # Step 3: Top-hat — bright structures (captures light reflex)
    top_hat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel)
    
    # Step 4: Bottom-hat — dark structures (captures vessel lumens)
    bottom_hat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)
    
    # Step 5: Subtract to suppress light reflex, enhance vessels
    # Use cv2.subtract to avoid uint8 underflow (clamps to 0)
    result = cv2.subtract(bottom_hat, top_hat)
    
    intermediates = {
        "green": green,
        "top_hat": top_hat,
        "bottom_hat": bottom_hat,
        "result": result
    }
    
    return result, intermediates

def visualize_phase1(img_rgb, intermediates):
    """Show all intermediate steps side by side for verification."""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original (RGB)")
    
    axes[1].imshow(intermediates["green"], cmap="gray")
    axes[1].set_title("Green Channel")
    
    axes[2].imshow(intermediates["top_hat"], cmap="gray")
    axes[2].set_title("Top-hat\n(light reflex captured)")
    
    axes[3].imshow(intermediates["bottom_hat"], cmap="gray")
    axes[3].set_title("Bottom-hat\n(dark vessels captured)")
    
    axes[4].imshow(intermediates["result"], cmap="gray")
    axes[4].set_title("T_output = BH - TH\n(light reflex suppressed)")
    
    for ax in axes:
        ax.axis("off")
    
    plt.suptitle("Phase 1: Pre-processing & Light Reflex Removal", fontsize=13)
    plt.tight_layout()
    plt.show()

# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    IMAGE_PATH = "test/21_training.tif"
    
    _, img_rgb = load_image(IMAGE_PATH)
    result, intermediates = phase1_preprocessing(img_rgb, kernel_radius=11)
    visualize_phase1(img_rgb, intermediates)
    
    print("Phase 1 complete. Shape:", result.shape, "| dtype:", result.dtype)