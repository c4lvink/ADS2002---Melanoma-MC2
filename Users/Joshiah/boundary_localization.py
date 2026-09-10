"""
Boundary localization (lesion cropping) -- reimplements Algorithm 1 from:

    Gajera, H.K., Nayak, D.R., Zaveri, M.A. (2023). "A comprehensive analysis
    of dermoscopy images for melanoma detection via deep CNN features."
    Biomedical Signal Processing and Control, 79, 104186.

Their algorithm takes a separately-provided binary lesion mask as input
(step 3: `mask <- imageread()`) and uses IT (not the original image) to find
the threshold and contour that get applied back onto the original image.
This project's dataset doesn't ship ground-truth lesion masks (only the
DullRazor hair mask exists, which marks hair strands, not the lesion), so
this version derives that segmentation mask directly from the image itself:
grayscale -> Gaussian blur -> Otsu threshold. That's the standard substitute
when no manual/ground-truth mask is available, and it's a reasonable one
here specifically because dermoscopy lesions are usually darker/more
saturated than the surrounding skin. Every other step (blur, threshold,
largest contour, bounding rectangle, crop) matches the paper's algorithm
directly.

Real dermoscopy images turned up failure modes plain Otsu-thresholding
doesn't handle on its own, each addressed below:
  1. Noisy border/vignette texture forming one connected region *larger*
     than the actual (roughly centered) lesion -> a morphological opening
     erodes away thin/small noise, a border margin is zeroed out, and
     contour selection requires candidates to be both large enough AND
     roughly centered, not just the single largest connected region.
  2. A wide flat band of border noise that's technically both "large" and
     "central" (its centroid sits near the image center even though the
     band itself hugs an edge) -> a circularity filter rejects long/thin
     shapes, kept lenient since lesion borders are often genuinely
     irregular (that irregularity is itself diagnostic).
  3. A thick uniform white (sticker-style) or black frame border around the
     actual photo -> left in, Otsu's *global* threshold ends up separating
     "border" vs "everything else" (skin+lesion together) instead of
     "skin" vs "lesion", so the whole inset photo becomes the mask. Fixed
     by trimming uniform border rows/columns before anything else runs.
  4. A dermatologist's skin-marker ink (purple/blue -- a ring around the
     lesion, reference dots/lines) is dark and high-contrast enough to win
     over the actual lesion under grayscale-only Otsu thresholding, which
     can't tell "dark ink" apart from "dark pigmented lesion". Fixed by
     remove_marker() below, which uses color (not just darkness) to find
     and inpaint out marker-colored pixels before anything else runs.
  5. Black/dark marker ink and pen writing -- color can't distinguish this
     from a dark lesion the way it can for purple/blue ink. Fixed by
     remove_dark_marker() below, which uses shape instead of color (the
     same blackhat technique remove_hair/DullRazor uses): pen strokes and
     letters are thin/elongated, a lesion blob is not.

Usage (after the notebook's DullRazor cells have populated train_hairless/):

    python boundary_localization.py --image-dir train_hairless --n-samples 6
    python boundary_localization.py --image-dir train_hairless --filenames ISIC_0343061.jpg
"""
import argparse
import glob
import os

import cv2
import numpy as np
from PIL import Image


def remove_marker(img_rgb, hue_low=100, hue_high=165, sat_thresh=40, val_thresh=200, inpaint_radius=6):
    """
    Removes surgical/dermatologist skin-marker ink (typically purple or blue
    -- a ring drawn around the lesion, reference dots, lines) before
    boundary localization runs. Left in, a dark, high-contrast marker mark
    can win over the actual lesion under Otsu thresholding, since plain
    grayscale intensity can't tell "dark ink" apart from "dark pigmented
    lesion" -- color can, though. Deliberately targets purple/blue hues
    only (not black ink): skin lesions are essentially never blue/purple
    (melanin gives brown/black/tan tones), so this is a low-false-positive
    signal, unlike trying to detect black ink, which would be very hard to
    tell apart from a genuinely dark lesion using color alone.

    Returns (cleaned_img_rgb, marker_mask) -- same img_rgb back unchanged
    if no marker-colored pixels were found.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    marker_mask = ((hue >= hue_low) & (hue <= hue_high) & (sat >= sat_thresh) & (val <= val_thresh))
    marker_mask = marker_mask.astype(np.uint8) * 255
    if not marker_mask.any():
        return img_rgb, marker_mask
    return cv2.inpaint(img_rgb, marker_mask, inpaint_radius, cv2.INPAINT_TELEA), marker_mask


def remove_dark_marker(img_rgb, kernel_size=21, thresh=30, inpaint_radius=6):
    """
    Removes black/dark marker ink and pen writing that remove_marker's
    color-based approach can't catch (it deliberately skips black ink,
    since color alone can't tell black ink apart from a dark lesion). This
    uses the same blackhat approach as remove_hair (DullRazor) instead --
    a blackhat filter highlights thin/elongated dark structures against
    lighter surroundings regardless of color, purely by shape, which is
    why it works for black ink specifically: pen strokes and handwritten
    letters are much thinner and more elongated than a lesion blob, so a
    kernel wide enough to enclose a stroke won't respond the same way to
    a solid lesion that's wider than the kernel.

    Uses a bigger kernel (21 vs remove_hair's 9) since pen strokes/letters
    are typically thicker than a single hair strand, and a higher contrast
    threshold (30 vs remove_hair's 10) -- tested empirically to still catch
    bold marker writing while mostly leaving a lesion's own irregular/
    jagged border alone (a real, if smaller, risk: a sharp concave edge
    feature on an irregular lesion can locally resemble a thin dark
    structure too, so this isn't perfectly risk-free, just tuned to
    minimize it).

    Returns (cleaned_img_rgb, dark_marker_mask) -- same img_rgb back
    unchanged if no qualifying dark structures were found.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, dark_marker_mask = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)
    if not dark_marker_mask.any():
        return img_rgb, dark_marker_mask
    return cv2.inpaint(img_rgb, dark_marker_mask, inpaint_radius, cv2.INPAINT_TELEA), dark_marker_mask


def boundary_localization_crop(
    image,
    blur_ksize=(15, 15),
    thresh_val=0,
    max_val=255,
    thresh_technique=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    morph_kernel_size=15,
    border_margin_frac=0.08,
    min_area_frac=0.01,
    central_frac=0.8,
    min_circularity=0.25,
    strip_uniform_border=True,
    uniform_border_std=5.0,
    uniform_border_bright=200,
    uniform_border_dark=20,
    strip_marker=True,
    marker_hue_low=100,
    marker_hue_high=165,
    marker_sat_thresh=40,
    marker_val_thresh=200,
    marker_inpaint_radius=6,
    strip_dark_marker=True,
    dark_marker_kernel_size=21,
    dark_marker_thresh=30,
    dark_marker_inpaint_radius=6,
    output_size=None,
):
    """
    Parameters
    ----------
    image : np.ndarray, RGB, shape (H, W, 3), dtype uint8
    blur_ksize : Gaussian blur kernel size (paper's step 5). Bumped from
        (5, 5) to (15, 15) -- a bigger blur smooths away small-scale skin
        texture/residual hair before thresholding, instead of letting it
        survive as its own little dark speck.
    thresh_val, max_val, thresh_technique : passed to cv2.threshold (step 6).
        Otsu's method (auto threshold) + THRESH_BINARY_INV, same reasoning
        as before -- lesions are darker than skin, INV makes the *lesion*
        the foreground.
    morph_kernel_size : size of the elliptical kernel used for a morphological
        "opening" (erode then dilate) followed by a "closing" applied to the
        mask before contour search. Opening erodes away anything thinner than
        the kernel -- thin leftover hair strands, speckle noise; closing fills
        small gaps back in so a real but slightly patchy lesion blob doesn't
        stay fragmented. Set to 0 to disable both.
    border_margin_frac : fraction of each (post-border-strip) edge to zero
        out in the mask before contour search. Dermoscopy images often have
        vignetting/uneven illumination right at the border, which can
        otherwise form a ring-shaped contour bigger than the actual lesion.
    min_area_frac, central_frac : among all contours found, only candidates
        that are at least `min_area_frac` of the image's area AND whose
        centroid falls within the central `central_frac` of each dimension
        are considered "plausible lesion" candidates.
    min_circularity : candidates also need circularity (4*pi*area/perimeter^2,
        1.0 for a perfect circle, near 0 for long/thin shapes) of at least
        this value. This is what catches the failure mode area+centrality
        alone didn't: a wide flat band of border noise can be both large
        AND technically "central" (its centroid sits near the image center
        even though the band itself hugs an edge), but it's never circular,
        so this filters it out. Kept deliberately lenient (0.25, not close
        to 1.0) because melanoma lesions are often genuinely irregular/
        asymmetric -- border irregularity is itself a diagnostic feature
        (the "B" in the ABCD rule) -- so demanding near-perfect circularity
        would systematically discard exactly the lesions most worth flagging.
        Among everything that survives both filters, the largest by area
        is picked, not the most circular -- circularity here is a filter
        against non-lesion shapes, not a target to maximize.
    strip_uniform_border, uniform_border_std, uniform_border_bright,
    uniform_border_dark : some images have a thick uniform white (sticker-
        style) or black frame border around the actual photo. Left in,
        Otsu's *global* threshold ends up separating "border" vs
        "everything else" (skin+lesion together) instead of "skin" vs
        "lesion" -- so the whole inset photo becomes the mask, not just the
        lesion. This trims rows/columns from each edge inward while they're
        near-uniform (std below `uniform_border_std`) AND near-white (mean
        above `uniform_border_bright`) or near-black (mean below
        `uniform_border_dark`), before anything else runs. Set
        strip_uniform_border=False to disable.
    strip_marker, marker_hue_low, marker_hue_high, marker_sat_thresh,
    marker_val_thresh, marker_inpaint_radius : see remove_marker() above --
        applied first, before border-stripping or thresholding, so a
        purple/blue marker mark can't be picked up as the "lesion" by
        Otsu thresholding (which only sees darkness, not color, so it
        can't otherwise tell ink apart from pigmented skin). Set
        strip_marker=False to disable.
    strip_dark_marker, dark_marker_kernel_size, dark_marker_thresh,
    dark_marker_inpaint_radius : see remove_dark_marker() above -- applied
        right after remove_marker, catches black/dark marker ink and
        writing that the color-based check above can't (it deliberately
        skips black ink). Set strip_dark_marker=False to disable.
    output_size : if given, the returned roi is resized to
        (output_size, output_size) via cv2.resize -- so every crop comes
        back at a consistent size (e.g. IMG_SIZE) instead of the raw,
        variable bounding-box size.

    Returns
    -------
    roi : np.ndarray, the cropped (and optionally resized) lesion region
    bbox : (x, y, w, h) the bounding box used to produce roi, in the
        *original* image's coordinates (before any resize, border strip,
        or marker removal)
    mask : np.ndarray, the binary mask used internally, after the
        morphological/border-margin cleanup (handy for debugging) -- sized
        to the post-border-strip content region, not the original image
    """
    if strip_marker:
        image, _ = remove_marker(image, marker_hue_low, marker_hue_high,
                                  marker_sat_thresh, marker_val_thresh, marker_inpaint_radius)

    if strip_dark_marker:
        image, _ = remove_dark_marker(image, dark_marker_kernel_size, dark_marker_thresh,
                                       dark_marker_inpaint_radius)

    h_img, w_img = image.shape[:2]
    gray_full = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    top, bottom, left, right = 0, h_img, 0, w_img
    if strip_uniform_border:
        def is_uniform(line):
            line = line.astype(np.float64)
            return line.std() < uniform_border_std and (
                line.mean() > uniform_border_bright or line.mean() < uniform_border_dark
            )
        while top < bottom - 1 and is_uniform(gray_full[top, left:right]):
            top += 1
        while bottom > top + 1 and is_uniform(gray_full[bottom - 1, left:right]):
            bottom -= 1
        while left < right - 1 and is_uniform(gray_full[top:bottom, left]):
            left += 1
        while right > left + 1 and is_uniform(gray_full[top:bottom, right - 1]):
            right -= 1

    content = image[top:bottom, left:right]                            # border stripped
    gray = gray_full[top:bottom, left:right]
    h_c, w_c = gray.shape

    blurred = cv2.GaussianBlur(gray, blur_ksize, 0)                    # step 5: blur

    _, mask = cv2.threshold(blurred, thresh_val, max_val, thresh_technique)  # step 6

    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # erode away thin noise/hair
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # then fill small gaps back in

    if border_margin_frac > 0:
        my, mx = int(h_c * border_margin_frac), int(w_c * border_margin_frac)
        if my > 0:
            mask[:my, :] = 0
            mask[-my:, :] = 0
        if mx > 0:
            mask[:, :mx] = 0
            mask[:, -mx:] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # step 7

    if not contours:
        # No lesion boundary found -- fall back to the (border-stripped) content
        # unmodified rather than crashing (the paper's algorithm doesn't cover this case).
        roi, bbox_local = content, (0, 0, w_c, h_c)
    else:
        content_area = h_c * w_c
        cx_c, cy_c = w_c / 2, h_c / 2

        def circularity(c):
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            return 0.0 if perimeter == 0 else 4 * np.pi * area / (perimeter ** 2)

        def is_plausible(c):
            if cv2.contourArea(c) < min_area_frac * content_area:
                return False
            if circularity(c) < min_circularity:
                return False
            x, y, w, h = cv2.boundingRect(c)
            ccx, ccy = x + w / 2, y + h / 2
            return (abs(ccx - cx_c) <= central_frac * w_c / 2
                    and abs(ccy - cy_c) <= central_frac * h_c / 2)

        candidates = [c for c in contours if is_plausible(c)] or contours  # step 8, filtered
        cnt_best = max(candidates, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(cnt_best)                        # steps 9-10
        roi, bbox_local = content[y : y + h, x : x + w], (x, y, w, h)  # step 11

    bx, by, bw, bh = bbox_local
    bbox = (left + bx, top + by, bw, bh)                               # back to original coords

    if output_size is not None:
        roi = cv2.resize(roi, (output_size, output_size), interpolation=cv2.INTER_AREA)

    return roi, bbox, mask


def _load_rgb(path):
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def demo(image_dir, n_samples=6, seed=0, filenames=None, save_path="boundary_localization_demo.png"):
    """
    Crops a handful of sample images from `image_dir` and plots
    original / mask / cropped side by side so the result can be eyeballed.

    Pass `filenames` (a list of basenames, e.g. ["ISIC_0343061.jpg"]) to
    check specific images instead of a random sample -- handy for
    re-checking a particular image you know was problematic.
    """
    import matplotlib.pyplot as plt

    if filenames:
        sample_paths = [os.path.join(image_dir, f) for f in filenames]
        missing = [p for p in sample_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"not found: {missing}")
    else:
        paths = sorted(
            p
            for p in glob.glob(os.path.join(image_dir, "*"))
            if p.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not paths:
            raise FileNotFoundError(f"no images found in {image_dir!r}")

        rng = np.random.default_rng(seed)
        n_samples = min(n_samples, len(paths))
        sample_paths = [paths[i] for i in rng.choice(len(paths), size=n_samples, replace=False)]

    fig, axes = plt.subplots(3, len(sample_paths), figsize=(3 * len(sample_paths), 9), squeeze=False)
    for col, path in enumerate(sample_paths):
        image = _load_rgb(path)
        roi, (x, y, w, h), mask = boundary_localization_crop(image)

        axes[0, col].imshow(image)
        axes[0, col].add_patch(
            plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2)
        )
        axes[0, col].set_title(os.path.basename(path), fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(mask, cmap="gray")
        axes[1, col].axis("off")

        axes[2, col].imshow(roi)
        axes[2, col].axis("off")

    axes[0, 0].text(-0.15, 0.5, "original\n+ bbox", transform=axes[0, 0].transAxes,
                     rotation=90, va="center", ha="center", fontsize=10)
    axes[1, 0].text(-0.15, 0.5, "mask used", transform=axes[1, 0].transAxes,
                     rotation=90, va="center", ha="center", fontsize=10)
    axes[2, 0].text(-0.15, 0.5, "cropped", transform=axes[2, 0].transAxes,
                     rotation=90, va="center", ha="center", fontsize=10)

    fig.suptitle("Boundary localization: original (with crop box) / mask / result")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", default="train_hairless",
                         help="directory of images to sample from (default: train_hairless)")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--filenames", nargs="+", default=None,
                         help="specific filenames to check instead of a random sample, "
                              "e.g. --filenames ISIC_0343061.jpg")
    parser.add_argument("--save-path", default="boundary_localization_demo.png")
    args = parser.parse_args()

    demo(args.image_dir, n_samples=args.n_samples, seed=args.seed,
         filenames=args.filenames, save_path=args.save_path)
