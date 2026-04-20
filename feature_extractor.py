import cv2
import numpy as np
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
from PIL import Image

FIXED_SIZE = (150, 150)

def safe_stat(func, arr, default=0.0):
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if arr.size == 0:
        return float(default)
    try:
        val = func(arr)
        if np.isnan(val) or np.isinf(val):
            return float(default)
        return float(val)
    except Exception:
        return float(default)


def read_image(uploaded_file):
    """Support Streamlit UploadedFile or PIL Image"""
    if isinstance(uploaded_file, Image.Image):
        img_bgr = cv2.cvtColor(np.array(uploaded_file), cv2.COLOR_RGB2BGR)
    else:
        bytes_data = uploaded_file.getvalue()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        raise ValueError("Could not read the uploaded image.")
    
    img_bgr = cv2.resize(img_bgr, FIXED_SIZE, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return img_rgb, img_gray, img_hsv, img_lab


def extract_color_features(img_rgb, img_gray, img_hsv, img_lab):
    feats = {}
    # RGB
    for i, ch_name in enumerate(['r', 'g', 'b']):
        ch = img_rgb[:, :, i].astype(np.float32)
        feats[f'rgb_mean_{ch_name}'] = float(np.mean(ch))
        feats[f'rgb_std_{ch_name}']  = float(np.std(ch))
        feats[f'rgb_skew_{ch_name}'] = safe_stat(skew, ch)

    # HSV
    for i, ch_name in enumerate(['h', 's', 'v']):
        ch = img_hsv[:, :, i].astype(np.float32)
        feats[f'hsv_mean_{ch_name}'] = float(np.mean(ch))
        feats[f'hsv_std_{ch_name}']  = float(np.std(ch))

    # LAB
    for i, ch_name in enumerate(['l', 'a', 'b_lab']):
        ch = img_lab[:, :, i].astype(np.float32)
        feats[f'lab_mean_{ch_name}'] = float(np.mean(ch))
        feats[f'lab_std_{ch_name}']  = float(np.std(ch))

    # Grayscale
    feats['gray_mean']     = float(np.mean(img_gray))
    feats['gray_std']      = float(np.std(img_gray))
    feats['gray_skew']     = safe_stat(skew, img_gray)
    feats['gray_kurtosis'] = safe_stat(kurtosis, img_gray)

    hist_density, _ = np.histogram(img_gray, bins=256, range=(0, 256), density=True)
    feats['gray_entropy'] = float(-np.sum(hist_density * np.log2(hist_density + 1e-12)))

    # RGB 8-bin histograms
    bins = 8
    for i, ch_name in enumerate(['r', 'g', 'b']):
        hist = cv2.calcHist([img_rgb], [i], None, [bins], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-12)
        for j, val in enumerate(hist):
            feats[f'rgb_hist_{ch_name}_{j}'] = float(val)

    # Color ratios + Vegetation + Gray
    r, g, b = img_rgb[:, :, 0].astype(float), img_rgb[:, :, 1].astype(float), img_rgb[:, :, 2].astype(float)
    total = r + g + b + 1e-12
    feats['green_ratio'] = float(np.mean(g / total))
    feats['red_ratio']   = float(np.mean(r / total))
    feats['blue_ratio']  = float(np.mean(b / total))

    lower_green = np.array([25, 20, 20], dtype=np.uint8)
    upper_green = np.array([100, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
    feats['vegetation_pixel_ratio'] = float(np.sum(green_mask > 0) / green_mask.size)

    s_channel = img_hsv[:, :, 1]
    feats['gray_pixel_ratio'] = float(np.sum(s_channel < 30) / s_channel.size)

    return feats


def extract_glcm_features(img_gray):
    feats = {}
    glcm = graycomatrix(img_gray, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    for prop in ['contrast', 'correlation', 'energy', 'homogeneity']:
        values = graycoprops(glcm, prop).flatten()
        feats[f'glcm_{prop}_mean'] = float(np.mean(values))
        feats[f'glcm_{prop}_std']  = float(np.std(values))
    return feats


def extract_lbp_features(img_gray, radius=1, n_points=8):
    feats = {}
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    for i, val in enumerate(hist):
        feats[f'lbp_hist_{i}'] = float(val)
    feats['lbp_mean'] = float(np.mean(lbp))
    feats['lbp_std']  = float(np.std(lbp))
    return feats


def extract_edge_features(img_gray):
    feats = {}
    edges = cv2.Canny(img_gray, 50, 150)
    feats['edge_density'] = float(np.sum(edges > 0) / edges.size)

    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    feats['grad_mean'] = float(np.mean(grad_mag))
    feats['grad_std']  = float(np.std(grad_mag))
    feats['grad_max']  = float(np.max(grad_mag))

    abs_gx = np.abs(grad_x)
    abs_gy = np.abs(grad_y)
    feats['vertical_edge_ratio']   = float(np.mean(abs_gx) / (np.mean(abs_gy) + 1e-12))
    feats['horizontal_edge_ratio'] = float(np.mean(abs_gy) / (np.mean(abs_gx) + 1e-12))

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=30, maxLineGap=10)
    feats['hough_line_count'] = float(len(lines) if lines is not None else 0)
    return feats


def extract_hog_features(img_gray):
    feats = {}
    try:
        hog_vec = hog(img_gray, orientations=9, pixels_per_cell=(16, 16),
                      cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        for i, val in enumerate(hog_vec):
            feats[f'hog_{i:03d}'] = float(val)
    except Exception:
        pass
    return feats


def extract_scene_structure_features(img_rgb, img_gray):
    feats = {}
    h, w = img_gray.shape

    upper_mean = float(np.mean(img_gray[:h//2, :]))
    lower_mean = float(np.mean(img_gray[h//2:, :]))
    feats['upper_lower_intensity_diff'] = upper_mean - lower_mean

    left  = img_gray[:, :w//2].astype(float)
    right = np.fliplr(img_gray[:, w//2:]).astype(float)
    min_w = min(left.shape[1], right.shape[1])
    diff = np.mean(np.abs(left[:, :min_w] - right[:, :min_w]))
    feats['lr_symmetry_score'] = float(1.0 - diff / 255.0)

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    upper_hsv = img_hsv[:h//3, :, :]
    lower_sky = np.array([90, 20, 100], dtype=np.uint8)
    upper_sky = np.array([130, 255, 255], dtype=np.uint8)
    sky_mask = cv2.inRange(upper_hsv, lower_sky, upper_sky)
    feats['sky_pixel_ratio'] = float(np.sum(sky_mask > 0) / sky_mask.size)

    patch_size = 32
    variances = []
    for row in range(0, h - patch_size, patch_size):
        for col in range(0, w - patch_size, patch_size):
            patch = img_gray[row:row+patch_size, col:col+patch_size]
            variances.append(float(np.var(patch)))

    if variances:
        feats['patch_variance_mean'] = float(np.mean(variances))
        feats['patch_variance_std']  = float(np.std(variances))
        feats['patch_regularity']    = float(1.0 / (np.std(variances) + 1e-12))
    else:
        feats['patch_variance_mean'] = 0.0
        feats['patch_variance_std']  = 0.0
        feats['patch_regularity']    = 0.0

    return feats


def extract_all_features(uploaded_file):
    """Main entry point for Streamlit app"""
    img_rgb, img_gray, img_hsv, img_lab = read_image(uploaded_file)
    features = {}
    features.update(extract_color_features(img_rgb, img_gray, img_hsv, img_lab))
    features.update(extract_glcm_features(img_gray))
    features.update(extract_lbp_features(img_gray))
    features.update(extract_edge_features(img_gray))
    features.update(extract_hog_features(img_gray))
    features.update(extract_scene_structure_features(img_rgb, img_gray))
    return features