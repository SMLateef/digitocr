# utils.py
import cv2
import numpy as np

def to_grayscale(img_bgr):
    if len(img_bgr.shape) == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_bgr

def denoise_and_threshold(gray):
    # gaussian blur -> Otsu threshold
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ensure digits are white (255) and background black (0) — MNIST style
    # if background is white and digit dark, invert
    if np.mean(th) > 127:
        th = 255 - th
    return th

def deskew(img_bin):
    # expects binary image (0/255), deskew using image moments
    img = img_bin.copy()
    img = img.astype(np.uint8)
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * img.shape[0] * skew],
                    [0, 1, 0]])
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def crop_to_bounding_box(img_bin, pad=2):
    # find bounding rect of non-zero pixels and crop with padding
    coords = cv2.findNonZero(img_bin)
    if coords is None:
        return img_bin
    x, y, w, h = cv2.boundingRect(coords)
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img_bin.shape[1])
    y2 = min(y + h + pad, img_bin.shape[0])
    return img_bin[y1:y2, x1:x2]

def resize_and_pad(img, size=28, pad_value=0):
    # expects binary or grayscale image with digit in white on black
    h, w = img.shape
    # if blank
    if h == 0 or w == 0:
        return np.full((size,size), pad_value, dtype=np.uint8)

    # scale to fit into 20x20 box (similar to MNIST preprocessing)
    max_side = max(h, w)
    scale = 20.0 / max_side
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # place centered into size x size
    padded = np.full((size, size), pad_value, dtype=np.uint8)
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # optional center of mass shift to center (improves MNIST-style performance)
    cy, cx = ndimage_center_of_mass(padded)
    shift_x = int(np.round(size/2 - cx))
    shift_y = int(np.round(size/2 - cy))
    M = np.float32([[1,0,shift_x],[0,1,shift_y]])
    shifted = cv2.warpAffine(padded, M, (size,size), borderValue=pad_value)
    return shifted

def ndimage_center_of_mass(img):
    # compute center of mass (x,y) of white pixels
    # return (cy, cx)
    h, w = img.shape
    Y, X = np.indices((h,w))
    img_f = img.astype(np.float32)/255.0
    total = img_f.sum()
    if total == 0:
        return (h/2, w/2)
    cx = (X * img_f).sum() / total
    cy = (Y * img_f).sum() / total
    return (cy, cx)

def preprocess_for_mnist(img_bgr):
    gray = to_grayscale(img_bgr)
    th = denoise_and_threshold(gray)
    th = deskew(th)
    cropped = crop_to_bounding_box(th)
    final = resize_and_pad(cropped, size=28, pad_value=0)
    # normalize float32 0..1 and invert to match training if needed
    return final.astype(np.float32)/255.0

def segment_digits_from_image(img_bgr, min_area=100):
    """Simple contour-based segmentation: returns list of cropped grayscale digit images."""
    gray = to_grayscale(img_bgr)
    th = denoise_and_threshold(gray)
    # find contours on thresh (digits white)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h < min_area:
            continue
        boxes.append((x,y,w,h))
    # sort left-to-right by x
    boxes = sorted(boxes, key=lambda b: b[0])
    crops = [th[y:y+h, x:x+w] for (x,y,w,h) in boxes]
    return crops, boxes
