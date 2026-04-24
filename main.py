import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import wiener
from skimage.filters import frangi

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
    """
    _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    kernel = get_disk_kernel(5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# =============================================================================
# PHASE 1
# =============================================================================
def phase1_preprocessing(img_rgb, kernel_radius=7):
    green = img_rgb[:, :, 1]
    retinal_mask = create_retinal_mask(green)
    kernel = get_disk_kernel(kernel_radius)

    top_hat    = cv2.morphologyEx(green, cv2.MORPH_TOPHAT,  kernel)
    bottom_hat = cv2.morphologyEx(green, cv2.MORPH_BLACKHAT, kernel)
    correction = cv2.subtract(bottom_hat, top_hat)
    corrected  = cv2.add(green, correction)
    corrected  = cv2.bitwise_and(corrected, corrected, mask=retinal_mask)

    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=retinal_mask)

    intermediates = {
        "green": green, "mask": retinal_mask,
        "top_hat": top_hat, "bottom_hat": bottom_hat,
        "correction": correction, "corrected": corrected,
        "result": enhanced
    }
    return enhanced, intermediates

def visualize_phase1(img_rgb, intermediates):
    fig, axes = plt.subplots(1, 6, figsize=(24, 4))
    axes[0].imshow(img_rgb);             axes[0].set_title("Original (RGB)")
    axes[1].imshow(intermediates["green"],      cmap="gray"); axes[1].set_title("Green Channel")
    axes[2].imshow(intermediates["mask"],       cmap="gray"); axes[2].set_title("Retinal Mask")
    axes[3].imshow(intermediates["correction"], cmap="gray"); axes[3].set_title("BH - TH Correction")
    axes[4].imshow(intermediates["corrected"],  cmap="gray"); axes[4].set_title("Green + Correction")
    axes[5].imshow(intermediates["result"],     cmap="gray"); axes[5].set_title("After CLAHE")
    for ax in axes: ax.axis("off")
    plt.suptitle("Phase 1: Pre-processing & Light Reflex Removal", fontsize=13)
    plt.tight_layout(); plt.show()

# =============================================================================
# PHASE 2
# =============================================================================
def homomorphic_filter(image, mask, sigma=30, gamma_low=0.5, gamma_high=2.0):
    img_float    = image.astype(np.float64)
    retinal_mean = img_float[mask > 0].mean()
    img_filled   = img_float.copy()
    img_filled[mask == 0] = retinal_mean
    img_filled   = img_filled + 1.0
    log_img      = np.log(img_filled)

    fft_shifted  = np.fft.fftshift(np.fft.fft2(log_img))
    rows, cols   = image.shape
    crow, ccol   = rows // 2, cols // 2
    u = np.arange(rows) - crow
    v = np.arange(cols) - ccol
    V, U         = np.meshgrid(v, u)
    H            = (gamma_high - gamma_low) * (1 - np.exp(-(U**2 + V**2) / (2 * sigma**2))) + gamma_low

    result_log   = np.real(np.fft.ifft2(np.fft.ifftshift(fft_shifted * H)))
    result       = np.exp(result_log) - 1.0
    result[mask == 0] = 0

    retinal_pixels = result[mask > 0]
    p1, p99        = np.percentile(retinal_pixels, 1), np.percentile(retinal_pixels, 99)
    clipped        = np.clip(retinal_pixels, p1, p99)
    result_norm    = np.zeros_like(result)
    result_norm[mask > 0] = (clipped - clipped.min()) / (clipped.max() - clipped.min()) * 255

    print(f"Homo post-norm: mean={result_norm[mask>0].mean():.1f} "
          f"min={result_norm[mask>0].min():.1f} max={result_norm[mask>0].max():.1f}")
    return result_norm.astype(np.uint8)

def wiener_filter(image, mask, window_size=5):
    img_float    = image.astype(np.float64)
    retinal_mean = img_float[mask > 0].mean()
    img_filled   = img_float.copy()
    img_filled[mask == 0] = retinal_mean

    flat_region  = (img_filled == retinal_mean)
    img_filled[flat_region] += np.random.uniform(0.1, 0.5, flat_region.sum())

    filtered = wiener(img_filled, mysize=window_size)
    filtered[mask == 0] = 0

    retinal_pixels = filtered[mask > 0]
    result_norm    = np.zeros_like(filtered)
    r_min, r_max   = retinal_pixels.min(), retinal_pixels.max()
    result_norm[mask > 0] = (retinal_pixels - r_min) / (r_max - r_min) * 255
    return result_norm.astype(np.uint8)

def phase2_enhancement(phase1_result, retinal_mask, sigma=30,
                        gamma_low=0.5, gamma_high=2.0, wiener_window=5):
    homo_result   = homomorphic_filter(phase1_result, retinal_mask, sigma, gamma_low, gamma_high)
    wiener_result = wiener_filter(homo_result, retinal_mask, wiener_window)
    return wiener_result, {"homo": homo_result, "wiener": wiener_result}

def visualize_phase2(phase1_result, intermediates):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(phase1_result,             cmap="gray"); axes[0].set_title("Phase 1 Output")
    axes[1].imshow(intermediates["homo"],     cmap="gray"); axes[1].set_title("After Homomorphic Filter")
    axes[2].imshow(intermediates["wiener"],   cmap="gray"); axes[2].set_title("After Wiener Filter")
    for ax in axes: ax.axis("off")
    plt.suptitle("Phase 2: Enhancement", fontsize=13)
    plt.tight_layout(); plt.show()

# =============================================================================
# PHASE 3  –  Multi-scale 2nd-order LoG  +  Anisotropic Diffusion
# =============================================================================

def build_log_kernels_batch(sigma_u_list, sigma_v, orientations, alpha=1.0, beta=0.5):
    """
    Pre-build all oriented 2nd-order Gaussian derivative kernels for one sigma_v.

    Returns a list of (kernel, sigma_u, theta) tuples so we can loop efficiently.
    Kernels are already scale-normalised and zero-meaned.
    """
    kernels = []
    for sigma_u in sigma_u_list:
        if sigma_u < 0.5:
            continue
        size = int(6 * max(sigma_u, sigma_v) + 1)
        if size % 2 == 0:
            size += 1
        half = size // 2
        y, x = np.mgrid[-half:half+1, -half:half+1]   # shape (size, size)

        for theta in orientations:
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            u =  x * cos_t + y * sin_t   # NOTE: paper convention
            v = -x * sin_t + y * cos_t

            gauss  = np.exp(-(u**2 / (2*sigma_u**2) + v**2 / (2*sigma_v**2)))
            kernel = ((u**2 - sigma_u**2) / (2 * np.pi * sigma_u**5 * sigma_v)) * gauss

            # Scale normalisation (Lindeberg)
            kernel *= (sigma_u ** alpha) * (sigma_v ** beta)

            # Zero-mean  → no DC offset
            kernel -= kernel.mean()
            kernels.append(kernel)
    return kernels


def multiscale_log_detector(image, mask, alpha=1.0, beta=0.5):
    from scipy.ndimage import convolve as ndconvolve

    # Paper parameters - DO NOT change these
    sigma_v_values     = [4, 5]
    elongation_factors = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    n_orientations     = 12
    orientations       = np.linspace(0, np.pi, n_orientations, endpoint=False)

    img_f32 = image.astype(np.float32) / 255.0

    retinal_mean = float(img_f32[mask > 0].mean())
    img_filled   = img_f32.copy()
    img_filled[mask == 0] = retinal_mean

    max_response = np.zeros_like(img_filled)

    total_kernels = len(sigma_v_values) * len(elongation_factors) * n_orientations
    done = 0

    for sigma_v in sigma_v_values:
        for f in elongation_factors:
            sigma_u = sigma_v * f
            if sigma_u < 0.5:
                continue

            size = int(6 * max(sigma_u, sigma_v) + 1)
            if size % 2 == 0:
                size += 1
            half = size // 2
            y, x = np.mgrid[-half:half+1, -half:half+1]

            for theta in orientations:
                cos_t, sin_t = np.cos(theta), np.sin(theta)

                # PAPER Eq 13 convention: u = x*cos - y*sin, v = x*sin + y*cos
                u = x * cos_t - y * sin_t
                v = x * sin_t + y * cos_t

                gauss  = np.exp(-(u**2 / (2*sigma_u**2) + v**2 / (2*sigma_v**2)))
                kernel = ((u**2 - sigma_u**2) / (2 * np.pi * sigma_u**5 * sigma_v)) * gauss

                # Scale normalisation (Lindeberg) - alpha=1, beta=0.5 per paper
                kernel *= (sigma_u ** alpha) * (sigma_v ** beta)
                kernel -= kernel.mean()  # zero-mean

                k32  = kernel.astype(np.float32)
                resp = ndconvolve(img_filled, k32, mode='reflect')

                # Dark vessels on bright background → positive response at vessel centre
                resp = np.maximum(resp, 0.0)
                np.maximum(max_response, resp, out=max_response)

                done += 1
                if done % 20 == 0:
                    print(f"  LoG kernels done: {done}/{total_kernels}")

    max_response[mask == 0] = 0.0

    retinal_px = max_response[mask > 0]
    print(f"LoG raw: mean={retinal_px.mean():.4f}  max={retinal_px.max():.4f}  "
          f"pixels>0: {np.sum(retinal_px>0)}  "
          f"pixels>p90: {np.sum(retinal_px > np.percentile(retinal_px,90))}")

    # Use p99.5 clip instead of p99 - vessels are in the very top of the distribution
    p0, p99 = np.percentile(retinal_px, 0), np.percentile(retinal_px, 99.5)
    clipped  = np.clip(retinal_px, p0, p99)

    result = np.zeros_like(max_response)
    rng    = clipped.max() - clipped.min()
    if rng > 0:
        result[mask > 0] = (clipped - clipped.min()) / rng * 255.0

    print(f"LoG result: mean={result[mask>0].mean():.1f}  "
          f"min={result[mask>0].min():.1f}  max={result[mask>0].max():.1f}")
    return result.astype(np.uint8)




def compute_entropy(image):
    """Spatial entropy of image — used as stopping criterion."""
    hist, _ = np.histogram(image, bins=256, range=(0, 256), density=True)
    hist     = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def anisotropic_diffusion(image, mask, max_iter=15, kappa=5, gamma=0.1):
    """
    Perona-Malik anisotropic diffusion with entropy-based stopping criterion.

    kappa  — edge sensitivity (lower = more edge preservation)
              Paper uses dark vessels; LoG output has BRIGHT vessels, so we
              want to smooth the background while keeping bright ridges → small κ
    gamma  — step size (must be < 0.25 for stability)
    """
    img = image.astype(np.float64)
    prev_entropy = compute_entropy(img)

    for i in range(max_iter):
        north = np.roll(img, -1, axis=0) - img
        south = np.roll(img,  1, axis=0) - img
        east  = np.roll(img, -1, axis=1) - img
        west  = np.roll(img,  1, axis=1) - img

        cn = np.exp(-(north/kappa)**2)
        cs = np.exp(-(south/kappa)**2)
        ce = np.exp(-(east /kappa)**2)
        cw = np.exp(-(west /kappa)**2)

        img = img + gamma * (cn*north + cs*south + ce*east + cw*west)
        img[mask == 0] = 0.0

        if i % 5 == 0 and i > 0:
            curr_entropy   = compute_entropy(img)
            entropy_change = abs(curr_entropy - prev_entropy)
            print(f"  Diffusion iter {i:3d}  entropy_change={entropy_change:.6f}")
            if entropy_change < 0.001:
                print(f"  Stopped at iteration {i}")
                break
            prev_entropy = curr_entropy

    retinal_px = img[mask > 0]
    result     = np.zeros_like(img)
    rng        = retinal_px.max() - retinal_px.min()
    if rng > 0:
        result[mask > 0] = (retinal_px - retinal_px.min()) / rng * 255.0

    print(f"Diffusion result: mean={result[mask>0].mean():.1f}  "
          f"min={result[mask>0].min():.1f}  max={result[mask>0].max():.1f}")
    return result.astype(np.uint8)


def phase3_vessel_coherence(phase2_intermediates, img_rgb, retinal_mask, alpha=1.0, beta=0.5):
    """
    Phase 3: Multi-scale LoG detection + Anisotropic Diffusion.
    
    PAPER FIX: The paper applies LoG to the GREEN CHANNEL of the Wiener-filtered
    image directly (Section 3.5, Fig 5→6), not to the chained homomorphic+wiener output.
    The Wiener is applied per RGB channel; green is selected for LoG.
    """
    # Get the green channel from original image and apply Wiener directly to it
    green_channel = img_rgb[:, :, 1]  # raw green channel
    
    # Apply Wiener filter directly to green channel (paper's pipeline: Wiener per channel → pick green)
    green_wiener = wiener_filter(green_channel, retinal_mask, window_size=3)
    
    print(f"Green-Wiener input to LoG: mean={green_wiener[retinal_mask>0].mean():.1f}")
    print("Running multi-scale LoG detector...")
    log_result = multiscale_log_detector(green_wiener, retinal_mask, alpha, beta)

    #Addition of novelty via Hessian-based Frangi filter fusion
    print("Running Frangi Vesselness filter...")
    frangi_img = frangi(green_wiener, sigmas = range(1,4,1), black_ridges = True)

    frangi_norm = np.zeros_like(frangi_img)
    frangi_norm = cv2.normalize(frangi_img, frangi_norm, 0, 255, cv2.NORM_MINMAX)
    frangi_norm = frangi_norm.astype(np.uint8)

    fused_result = np.maximum(log_result, frangi_norm)

    print("Running anisotropic diffusion...")
    diffusion_result = anisotropic_diffusion(fused_result, retinal_mask)

    intermediates = {
        "green_wiener": green_wiener,
        "frangi": frangi_norm,
        "log": log_result,
        "fused" : fused_result,
        "diffusion": diffusion_result
    }
    return diffusion_result, intermediates


def visualize_phase3(phase2_result, intermediates):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(intermediates["log"], cmap="gray")
    axes[0].set_title("1. Original LoG\n(Misses capillaries)")
    
    axes[1].imshow(intermediates["frangi"], cmap="gray")
    axes[1].set_title("2. Frangi Filter\n(Finds tiny tubes)")
    
    axes[2].imshow(intermediates["fused"], cmap="gray")
    axes[2].set_title("3. FUSED Features\n(Novelty!)")
    
    axes[3].imshow(intermediates["diffusion"], cmap="gray")
    axes[3].set_title("4. After Diffusion")
    
    for ax in axes: ax.axis("off")
    plt.suptitle("Phase 3: Hybrid Hessian-Laplacian Vessel Coherence", fontsize=15)
    plt.tight_layout()
    plt.show()

#Phase 4AND5

def morphological_reconstruction(marker, mask):
    """
    Morphological reconstruction by dilation. It will iteratively dilate the marker and mask until stability is reached.
    """
    kernel = np.ones((3, 3), np.uint8)
    curr_marker = marker.copy()

    while True:
        expanded = cv2.dilate(curr_marker, kernel, iterations = 1)
        expanded = cv2.bitwise_and(expanded, mask)

        #Image change check
        if np.array_equal(curr_marker, expanded):
            break
        curr_marker = expanded

    return curr_marker

def remove_small_objects(binary_image, min_size = 70):
    """
    As specified in the paper, remove connected components smaller than 70 pixels.
    """

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    cleaned_image = np.zeros_like(binary_image)

    for i in range(1, num_labels):
        if stats [i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_image[labels == i] = 255

    return cleaned_image

def phase4_5_segmentation(phase3_result, retinal_mask):
    """
    Phase 4: Double Thresholding using histogram and edge-based means.
    Phase 5: Morphological Reconstruction and small area removal.
    """

    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    safe_mask = cv2.erode(retinal_mask, erosion_kernel, iterations=1)

    retinal_pixels = phase3_result[safe_mask > 0]

    mean_val = np.mean(retinal_pixels)
    std_val = np.std(retinal_pixels)

    #Lower threshold
    T_L = mean_val + (0.7 * std_val)

    median_val = np.median(retinal_pixels)
    lower_canny = int(max(0, 0.66 * median_val))
    upper_canny = int(min(255, 1.33 * median_val))

    #Upper threshold
    edges = cv2.Canny(phase3_result, lower_canny, upper_canny)
    edge_pixels = phase3_result[(edges > 0) & (safe_mask > 0)]

    if len(edge_pixels) > 0:
        T_U = np.mean(edge_pixels)
    else:
        T_U = mean_val + (1.2 * std_val)

    if T_L > T_U:
        T_L, T_U = T_U, T_L

    print(f"Thresholds -> T_L: {T_L:.2f}, T_U: {T_U:.2f}")

    #Generate MASK and MARKER images
    _, mask_img = cv2.threshold(phase3_result, T_L, 255, cv2.THRESH_BINARY)
    _, marker_img = cv2.threshold(phase3_result, T_U, 255, cv2.THRESH_BINARY)

    #Constrain to retinal region
    mask_img = cv2.bitwise_and(mask_img, mask_img, mask=safe_mask)
    marker_img = cv2.bitwise_and(marker_img, marker_img, mask=safe_mask)

    print("Running morphological reconstruction...")
    reconstructed_img = morphological_reconstruction(marker_img, mask_img)

    print("Removing small noisy components...")
    final_segmented = remove_small_objects(reconstructed_img, min_size=70)

    intermediates = {
        "mask_img" : mask_img,
        "marker_img": marker_img,
        "reconstructed": reconstructed_img,
        "final": final_segmented
    }

    return final_segmented, intermediates

def visualize_phase4_5(intermediates):
    fig, axes = plt.subplots(2, 2, figsize=(12,12))
    axes[0,0].imshow(intermediates["mask_img"], cmap = "gray"); axes[0,0].set_title("Mask Image (T_L)")
    axes[0, 1].imshow(intermediates["marker_img"], cmap="gray"); axes[0, 1].set_title("Marker Image (T_U)")
    axes[1, 0].imshow(intermediates["reconstructed"], cmap="gray"); axes[1, 0].set_title("Morphological Reconstruction")
    axes[1, 1].imshow(intermediates["final"], cmap="gray"); axes[1, 1].set_title("Final Segmented (<70px removed)")

    for ax in axes.flat:
        ax.axis("off")

    plt.suptitle("Phase 4 & 5: Binarization and Reconstruction", fontsize=15)
    plt.tight_layout()
    plt.show()

#Evaluation

def evaluate_segmentation(predicted_img, ground_truth_img, mask_img):
    """
    Calculates Sensitivity, Specificity, Accuracy, and AUC within the retinal field of view
    from scratch, without using external machine learning libraries.
    """
    # 1. Convert images to strict binary arrays (0 and 1)
    pred_bin = (predicted_img > 127).astype(np.uint8)
    gt_bin = (ground_truth_img > 127).astype(np.uint8)
    
    # 2. Extract only the pixels inside the retina (ignoring the black corners)
    roi = mask_img > 0
    pred_roi = pred_bin[roi]
    gt_roi = gt_bin[roi]
    
    # 3. Calculate Confusion Matrix components
    # TP: Predicted is Vessel (1), Ground Truth is Vessel (1)
    TP = np.sum((pred_roi == 1) & (gt_roi == 1))
    
    # TN: Predicted is Background (0), Ground Truth is Background (0)
    TN = np.sum((pred_roi == 0) & (gt_roi == 0))
    
    # FP: Predicted is Vessel (1), Ground Truth is Background (0)
    FP = np.sum((pred_roi == 1) & (gt_roi == 0))
    
    # FN: Predicted is Background (0), Ground Truth is Vessel (1)
    FN = np.sum((pred_roi == 0) & (gt_roi == 1))
    
    # 4. Calculate Metrics
    # Avoid division by zero
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    # The paper explicitly states: "The AUC is calculated using the formula AUC=(Se+Sp)/2"
    auc = (sensitivity + specificity) / 2
    
    return {
        "Accuracy (AC)": accuracy,
        "Sensitivity (Se)": sensitivity,
        "Specificity (Sp)": specificity,
        "AUC": auc,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN
    }

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    IMAGE_PATH = "test/21_training.tif"
    GROUND_TRUTH_PATH = "test/21_manual1.gif"

    _, img_rgb = load_image(IMAGE_PATH)

    try:
        gt_img = np.array(Image.open(GROUND_TRUTH_PATH).convert('L'))
    except FileNotFoundError:
        print(f"Ground truth not found at {GROUND_TRUTH_PATH}. Evaluation will be skipped.")
        gt_img = None

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    p1_result, p1_intermediates = phase1_preprocessing(img_rgb, kernel_radius=7)
    visualize_phase1(img_rgb, p1_intermediates)
    retinal_mask = p1_intermediates["mask"]
    print(f"Phase1 mean (retinal): {p1_result[retinal_mask>0].mean():.2f}")

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    p2_result, p2_intermediates = phase2_enhancement(
        p1_result, retinal_mask,
        sigma=15, gamma_low=0.3, gamma_high=2.5, wiener_window=3
    )
    visualize_phase2(p1_result, p2_intermediates)
    print(f"Phase2 mean (retinal): {p2_result[retinal_mask>0].mean():.2f}")

    # ── Phase 3 ──────────────────────────────────────────────────────────────
    p3_result, p3_intermediates = phase3_vessel_coherence(
        p2_intermediates, img_rgb, retinal_mask, alpha=1.0, beta=0.5
    )
    visualize_phase3(p2_result, p3_intermediates)
    print(f"Phase3 mean (retinal): {p3_result[retinal_mask>0].mean():.2f}")

    # ── Phase 4 & 5 ──────────────────────────────────────────────────────────────
    final_result, p4_intermediates = phase4_5_segmentation(p3_result, retinal_mask)
    visualize_phase4_5(p4_intermediates)
    print("Segmentation complete.")

    # ── Phase 6: Evaluation ──────────────────────────────────────────────────
    if gt_img is not None:
        print("\n--- Evaluation Metrics ---")
        metrics = evaluate_segmentation(final_result, gt_img, retinal_mask)
        print(f"Accuracy (AC):    {metrics['Accuracy (AC)']:.4f}")
        print(f"Sensitivity (Se): {metrics['Sensitivity (Se)']:.4f}")
        print(f"Specificity (Sp): {metrics['Specificity (Sp)']:.4f}")
        print(f"AUC:              {metrics['AUC']:.4f}")
        
        # Visualize Prediction vs Ground Truth
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(final_result, cmap='gray'); axes[0].set_title("Predicted Vessels")
        axes[1].imshow(gt_img, cmap='gray'); axes[1].set_title("Ground Truth (Manual)")
        for ax in axes: ax.axis('off')
        plt.suptitle("Final Output Comparison")
        plt.tight_layout()
        plt.show()