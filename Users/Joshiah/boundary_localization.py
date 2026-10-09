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


def is_rectangular_contour(c, max_vertices=5, min_extent=0.8):
    """
    True when a contour looks like a man-made rectangular object (a calibration
    card, ruler, sticker label) rather than an organic lesion boundary. Circularity
    alone can't tell these apart -- a filled square already scores ~0.79 on
    4*pi*area/perimeter^2, comfortably past the lenient min_circularity cutoff used
    elsewhere in this file -- so this checks shape more directly, two ways at once:
    cv2.approxPolyDP at a fine tolerance (2% of perimeter) collapses a true straight-
    edged rectangle to about 4 vertices, while a real lesion boundary needs many more
    to trace its natural irregularity even when fairly round; and cv2.minAreaRect
    gives the contour's own rotated bounding rectangle, which a true rectangle fills
    almost completely (extent close to 1.0) but an organic blob typically fills well
    under min_extent of. Both must hold at once, so a small, simple, fairly round
    lesion (few vertices at this tolerance, but with plenty of empty space around it
    in its bounding rect) is not caught by this.
    """
    perimeter = cv2.arcLength(c, True)
    if perimeter == 0:
        return False
    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    (rect_w, rect_h) = cv2.minAreaRect(c)[1]
    if rect_w * rect_h == 0:
        return False
    extent = cv2.contourArea(c) / (rect_w * rect_h)
    return len(approx) <= max_vertices and extent >= min_extent


def strip_rectangular_artifacts(img_rgb, min_area_frac=0.02, max_iterations=3,
                                 blur_ksize=(15, 15), morph_kernel_size=15,
                                 rect_max_vertices=5, rect_min_extent=0.8):
    """
    Removes man-made rectangular objects (calibration cards, rulers, sticker labels)
    that survive as one large, high-contrast blob under the *same* Otsu threshold
    boundary_localization_crop's own lesion search uses further down -- rather than
    trying to darkness-threshold them away with a separate, fixed, hand-picked cutoff
    the way remove_border_vignette does. That fixed-threshold approach turned out to
    be resolution-fragile: at low resolution (e.g. this dataset's 512x512 images) a
    dark rectangular object's edge, where it blends into lighter skin, occupies a much
    bigger fraction of its silhouette than at high resolution, and a large chunk of
    the object ends up in a "medium gray" band that no single fixed threshold reaches
    -- so only a small fragment of it would get flagged and removed, leaving a dark,
    still-rectangular residual that keeps winning the contour search over the actual
    lesion. Otsu's threshold is chosen per-image instead of fixed, so it adapts to
    wherever that image's actual card/skin brightness gap falls, and reliably captures
    the whole object as a single clean contour in one pass.

    Iterates (up to max_iterations times) rather than running once: finds the single
    largest Otsu-thresholded contour, and if it is both large (>= min_area_frac of the
    image) and rectangular (is_rectangular_contour, using rect_max_vertices/
    rect_min_extent), fills it with the mean color of everything else and repeats --
    stopping as soon as the largest remaining contour is too small or not rectangular
    (a real lesion, even a large dark one, is never mistaken for "rectangular" by
    is_rectangular_contour, so this does not erode into it). Handles the rare case of
    more than one such object without needing a fixed count in advance.

    Returns the cleaned img_rgb (unchanged if nothing qualified).
    """
    image = img_rgb.copy()
    for _ in range(max_iterations):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            break
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) < min_area_frac * gray.size:
            break
        if not is_rectangular_contour(biggest, rect_max_vertices, rect_min_extent):
            break
        artifact_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(artifact_mask, [biggest], -1, 255, thickness=cv2.FILLED)
        non_artifact = artifact_mask == 0
        if not non_artifact.any():
            break
        fill_color = image[non_artifact].reshape(-1, 3).mean(axis=0)
        image[artifact_mask == 255] = fill_color
    return image


def remove_border_vignette(img_rgb, dark_thresh=60, inpaint_radius=6, morph_kernel_size=31):
    """
    Removes a dark border/vignette that touches the image edge -- e.g. a thick
    black surround around a smaller circular photographed area, common in ISIC
    images -- before boundary localization runs. Left in, this can otherwise
    out-compete the actual lesion: strip_uniform_border only catches borders
    that are near-perfectly uniform row-by-row/column-by-column (std below
    uniform_border_std), which a vignette with any gradient, compression noise,
    or non-rectangular (e.g. circular) shape won't pass; border_margin_frac only
    zeroes a fixed-width band, which a thicker or irregular vignette extends
    well past. Unlike either, this works for any shape or thickness, since it
    doesn't assume uniformity or a fixed width -- only that a vignette (unlike a
    lesion, which dermoscopy protocol keeps centered) touches the image border.

    Thresholds the image for dark pixels, then keeps only the connected dark
    regions that touch row/column 0 or the last row/column -- a real lesion,
    even a dark one, is essentially never connected all the way out to the
    image edge, so this is a low-false-positive way to isolate vignette/frame
    artifacts specifically, without also catching the lesion itself.

    Real borders are rarely perfectly solid -- compression noise, a slight
    brightness gradient, or small artifacts leave scattered pixels just above
    `dark_thresh`, punching tiny holes through the border in the raw dark
    mask. Those holes can locally break the border's connectivity to the
    image edge, so a chunk of border on the far side of a hole never gets
    flagged as "touching the border" and survives unfilled. `morph_kernel_size`
    runs a morphological closing on the dark mask first (dilate then erode,
    same technique boundary_localization_crop's own morph_kernel_size uses on
    its threshold mask) to bridge those small gaps before labeling connected
    components, so a border with minor internal noise still reads as one
    solid, edge-connected region. Set to 0 to disable.

    Returns (cleaned_img_rgb, vignette_mask) -- same img_rgb back unchanged if
    no border-touching dark region was found.

    Fills the vignette with a flat color (the mean of the surrounding
    non-vignette pixels) instead of cv2.inpaint()'s texture reconstruction --
    inpaint() (Telea) works well for remove_marker/remove_dark_marker's thin
    marks, but over a mask this large (a big chunk of the whole image), it has
    almost no real texture to propagate from and tends to wash out to a flat,
    near-white fill instead. That fill is *different* from the true
    surrounding skin tone, which can itself become a new spurious contour
    candidate downstream -- a flat fill matching the actual skin tone doesn't
    have that problem, and it doesn't need to look realistic since this
    region gets discarded by the crop either way.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    dark_mask = (gray < dark_thresh).astype(np.uint8)

    if morph_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)  # bridge small noise holes

    n_labels, labels = cv2.connectedComponents(dark_mask)

    border_labels = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    border_labels.discard(0)  # label 0 is the non-dark background, not a component

    if not border_labels:
        return img_rgb, np.zeros_like(dark_mask, dtype=np.uint8)

    vignette_mask = np.isin(labels, list(border_labels)).astype(np.uint8) * 255

    non_vignette = vignette_mask == 0
    if not non_vignette.any():
        # degenerate case: the whole image was flagged as vignette -- nothing
        # to average, so fall back to inpaint rather than filling with nothing
        return cv2.inpaint(img_rgb, vignette_mask, inpaint_radius, cv2.INPAINT_TELEA), vignette_mask

    fill_color = img_rgb[non_vignette].reshape(-1, 3).mean(axis=0)
    cleaned = img_rgb.copy()
    cleaned[vignette_mask == 255] = fill_color
    return cleaned, vignette_mask


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
    reject_rectangular=True,
    rect_max_vertices=5,
    rect_min_extent=0.8,
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
    strip_rectangular=True,
    rect_artifact_min_area_frac=0.02,
    rect_artifact_max_iterations=3,
    strip_border_vignette=True,
    vignette_dark_thresh=60,
    vignette_inpaint_radius=6,
    vignette_morph_kernel_size=31,
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
    strip_rectangular, rect_artifact_min_area_frac, rect_artifact_max_iterations,
    rect_max_vertices, rect_min_extent : man-made rectangular objects that end up in
        frame (a calibration card, ruler, sticker label) are dark/high-contrast enough
        to survive the filters above (min_area_frac, min_circularity, since a filled
        rectangle is reasonably circular too -- a square scores ~0.79, well clear of
        the lenient 0.25 cutoff). See is_rectangular_contour() and
        strip_rectangular_artifacts() above for the shape test and removal step this
        runs before anything else here -- deliberately before strip_border_vignette
        and strip_uniform_border, since a large rectangular artifact's darkness would
        otherwise skew this function's own Otsu threshold at the end. rect_max_vertices
        and rect_min_extent tune the same underlying shape test used a second time
        below, as a final safety net at contour-selection time (reject_rectangular) --
        in case a rectangular object is too small or too low-contrast for
        strip_rectangular_artifacts to have caught upstream, but still ends up the
        largest contour here. Set strip_rectangular=False or reject_rectangular=False
        to disable either independently.
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
    strip_border_vignette, vignette_dark_thresh, vignette_inpaint_radius,
    vignette_morph_kernel_size : see remove_border_vignette() above --
        applied right after remove_dark_marker, catches a dark border/
        vignette touching the image edge (any shape/thickness) that
        strip_uniform_border and border_margin_frac below can each miss on
        their own. Set strip_border_vignette=False to disable.
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

    if strip_rectangular:
        image = strip_rectangular_artifacts(image, rect_artifact_min_area_frac, rect_artifact_max_iterations,
                                             blur_ksize, morph_kernel_size, rect_max_vertices, rect_min_extent)

    if strip_border_vignette:
        image, _ = remove_border_vignette(image, vignette_dark_thresh, vignette_inpaint_radius, vignette_morph_kernel_size)

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
            if reject_rectangular and is_rectangular_contour(c, rect_max_vertices, rect_min_extent):
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
