import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import wiener

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
def create_retinal_mask(image, threshold=10):
    """
    Creates a binary mask of the actual retinal region,
    ignoring the black circular border in DRIVE images.
    threshold: pixels below this are considered border/background
    """
    _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    # clean up small noise in the mask
    kernel = get_disk_kernel(5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def phase1_preprocessing(img_rgb, kernel_radius=7):
    green = img_rgb[:, :, 1]
    retinal_mask = create_retinal_mask(green)
    kernel = get_disk_kernel(kernel_radius)
    
    top_hat = cv2.morphologyEx(green, cv2.MORPH_TOPHAT, kernel)
    bottom_hat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)
    correction = cv2.subtract(bottom_hat, top_hat)
    
    # Apply correction to green channel
    corrected = cv2.add(green, correction)
    
    # Apply mask
    corrected = cv2.bitwise_and(corrected, corrected, mask=retinal_mask)
    
    # CLAHE: enhances local contrast so vessels become clearly visible
    # clipLimit controls contrast enhancement strength
    # tileGridSize controls locality of enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)
    
    # Mask again after CLAHE
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=retinal_mask)
    
    intermediates = {
        "green": green,
        "mask": retinal_mask,
        "top_hat": top_hat,
        "bottom_hat": bottom_hat,
        "correction": correction,
        "corrected": corrected,
        "result": enhanced
    }
    
    return enhanced, intermediates

def visualize_phase1(img_rgb, intermediates):
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original (RGB)")
    
    axes[1].imshow(intermediates["green"], cmap="gray")
    axes[1].set_title("Green Channel")
    
    axes[2].imshow(intermediates["mask"], cmap="gray")
    axes[2].set_title("Retinal Mask")
    
    axes[3].imshow(intermediates["correction"], cmap="gray")
    axes[3].set_title("BH - TH Correction")
    
    axes[4].imshow(intermediates["corrected"], cmap="gray")
    axes[4].set_title("Green + Correction")
    
    axes[5].imshow(intermediates["result"], cmap="gray")
    axes[5].set_title("After CLAHE\n(vessels enhanced)")
    
    for ax in axes:
        ax.axis("off")
    
    plt.suptitle("Phase 1: Pre-processing & Light Reflex Removal", fontsize=13)
    plt.tight_layout()
    plt.show()


def homomorphic_filter(image, mask, sigma=30, gamma_low=0.5, gamma_high=2.0):
    """
    Before FFT, fill black border with local mean so zeros
    don't corrupt the frequency domain operations.
    """
    img_float = image.astype(np.float64)
    
    # Fill border region with mean of retinal pixels only
    retinal_mean = img_float[mask > 0].mean()
    img_filled = img_float.copy()
    img_filled[mask == 0] = retinal_mean  # border gets neutral value
    
    img_filled = img_filled + 1.0  # avoid log(0)
    log_img = np.log(img_filled)
    
    fft_img = np.fft.fft2(log_img)
    fft_shifted = np.fft.fftshift(fft_img)
    
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    V, U = np.meshgrid(v, u)
    D_squared = U**2 + V**2
    H = (gamma_high - gamma_low) * (1 - np.exp(-D_squared / (2 * sigma**2))) + gamma_low
    
    filtered_fft = fft_shifted * H
    ifft_shifted = np.fft.ifftshift(filtered_fft)
    result_log = np.real(np.fft.ifft2(ifft_shifted))
    result = np.exp(result_log) - 1.0
    
    # Mask border back to zero before normalizing
    result[mask == 0] = 0
    
    # Normalize only within retinal region
    retinal_pixels = result[mask > 0]
    p1, p99 = np.percentile(retinal_pixels, 1), np.percentile(retinal_pixels, 99)
    retinal_pixels_clipped = np.clip(retinal_pixels, p1, p99)
    
    result_norm = np.zeros_like(result)
    retinal_min = retinal_pixels_clipped.min()
    retinal_max = retinal_pixels_clipped.max()
    result_norm[mask > 0] = (retinal_pixels_clipped - retinal_min) / (retinal_max - retinal_min) * 255
    
    print(f"Homo post-norm: mean={result_norm[mask>0].mean():.1f} "
          f"min={result_norm[mask>0].min():.1f} "
          f"max={result_norm[mask>0].max():.1f}")
    
    return result_norm.astype(np.uint8)


def wiener_filter(image, mask, window_size=5):
    img_float = image.astype(np.float64)
    
    # Fill border with retinal mean before wiener sees it
    retinal_mean = img_float[mask > 0].mean()
    img_filled = img_float.copy()
    img_filled[mask == 0] = retinal_mean
    
    # Add small noise to flat regions to prevent zero variance
    flat_region = (img_filled == retinal_mean)
    img_filled[flat_region] += np.random.uniform(0.1, 0.5, flat_region.sum())
    
    filtered = wiener(img_filled, mysize=window_size)
    
    # Restore border to zero
    filtered[mask == 0] = 0
    
    retinal_pixels = filtered[mask > 0]
    result_norm = np.zeros_like(filtered)
    r_min, r_max = retinal_pixels.min(), retinal_pixels.max()
    result_norm[mask > 0] = (retinal_pixels - r_min) / (r_max - r_min) * 255
    
    return result_norm.astype(np.uint8)


def phase2_enhancement(phase1_result, retinal_mask, sigma=30, 
                        gamma_low=0.5, gamma_high=2.0, wiener_window=5):
    homo_result = homomorphic_filter(phase1_result, retinal_mask, 
                                      sigma, gamma_low, gamma_high)
    wiener_result = wiener_filter(homo_result, retinal_mask, wiener_window)
    
    intermediates = {
        "homo": homo_result,
        "wiener": wiener_result
    }
    return wiener_result, intermediates


def visualize_phase2(phase1_result, intermediates):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].imshow(phase1_result, cmap="gray")
    axes[0].set_title("Phase 1 Output")
    
    axes[1].imshow(intermediates["homo"], cmap="gray")
    axes[1].set_title("After Homomorphic Filter")
    
    axes[2].imshow(intermediates["wiener"], cmap="gray")
    axes[2].set_title("After Wiener Filter")
    
    for ax in axes:
        ax.axis("off")
    
    plt.suptitle("Phase 2: Enhancement", fontsize=13)
    plt.tight_layout()
    plt.show()

    # TEMP: histogram debug
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 3))
    axes2[0].hist(phase1_result.ravel(), bins=50, color='blue')
    axes2[0].set_title("Phase 1 histogram")
    axes2[1].hist(intermediates["homo"].ravel(), bins=50, color='green')
    axes2[1].set_title("Homo histogram")
    axes2[2].hist(intermediates["wiener"].ravel(), bins=50, color='red')
    axes2[2].set_title("Wiener histogram")
    plt.tight_layout()
    plt.show()

# ── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    IMAGE_PATH = "test/21_training.tif"
    
    _, img_rgb = load_image(IMAGE_PATH)
    
    # Phase 1
    p1_result, p1_intermediates = phase1_preprocessing(img_rgb, kernel_radius=7)
    visualize_phase1(img_rgb, p1_intermediates)
    
    retinal_mask = p1_intermediates["mask"]
    print(f"Green channel mean: {img_rgb[:,:,1][retinal_mask>0].mean():.2f}")
    print(f"Phase1 result mean (retinal only): {p1_result[retinal_mask>0].mean():.2f}")
    
    # Phase 2
   # Phase 2 - more aggressive high frequency boost
    p2_result, p2_intermediates = phase2_enhancement(
        p1_result, retinal_mask,
        sigma=15,        # tighter high-pass
        gamma_low=0.3,   # strongly suppress low freq illumination
        gamma_high=2.5,  # strongly boost high freq vessel detail
        wiener_window=3  # smaller window = less blurring
    )
    visualize_phase2(p1_result, p2_intermediates)
    print(f"Phase2 result mean (retinal only): {p2_result[retinal_mask>0].mean():.2f}")