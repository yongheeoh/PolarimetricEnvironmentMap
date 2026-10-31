"""Image-derived planar Floor registration, independent of scene orchestration."""

import cv2
import numpy as np
from pemap.geometry.projection import feature_image, transform


def register_floor(reference, known, image, valid):
    """Planar registration with an independent split-fit reprojection check."""
    sift = cv2.SIFT_create(nfeatures=18000, contrastThreshold=0.01, edgeThreshold=12)
    a, b = feature_image(image, valid), feature_image(reference, known)
    k1, d1 = sift.detectAndCompute(a, cv2.erode(binary(valid), np.ones((5, 5), np.uint8)))
    k2, d2 = sift.detectAndCompute(b, cv2.erode(binary(known), np.ones((9, 9), np.uint8)))
    report = {"accepted": False, "matches": 0}
    if d1 is None or d2 is None or min(len(d1), len(d2)) < 2:
        return report
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward, backward = matcher.knnMatch(d1, d2, k=2), matcher.knnMatch(d2, d1, k=2)
    back = {p[0].queryIdx: p[0].trainIdx for p in backward if len(p) == 2 and p[0].distance < 0.78 * p[1].distance}
    matches = [p[0] for p in forward if len(p) == 2 and p[0].distance < 0.78 * p[1].distance and back.get(p[0].trainIdx) == p[0].queryIdx]
    report["matches"] = len(matches)
    if len(matches) < 30:
        return report
    src = np.float32([k1[m.queryIdx].pt for m in matches])
    dst = np.float32([k2[m.trainIdx].pt for m in matches])
    order = np.random.default_rng(20261018).permutation(len(matches))
    test, train = order[::5], np.delete(order, np.arange(0, len(order), 5))
    cv2.setRNGSeed(20261018)
    held, _ = cv2.findHomography(src[train], dst[train], cv2.USAC_MAGSAC, 3.5, maxIters=20000, confidence=0.999)
    matrix, inliers = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.5, maxIters=20000, confidence=0.999)
    if held is None or matrix is None:
        return report
    good = inliers.ravel().astype(bool)
    sel = good[test]
    error = np.linalg.norm(transform(src, matrix) - dst, axis=1)
    held_error = np.linalg.norm(transform(src[test], held) - dst[test], axis=1)
    median = float(np.median(held_error[sel])) if sel.any() else None
    report.update(
        accepted=bool(good.sum() >= 30 and sel.sum() >= 5 and median < 4),
        inliers=int(good.sum()),
        heldout_inlier_count=int(sel.sum()),
        heldout_median_px=median,
        median_reprojection_px=float(np.median(error[good])),
        homography_source_to_nadir=matrix.tolist(),
        validation_note="20% held out from split fit; reported correspondences filtered by the all-match robust fit",
    )
    return report


from pemap.io.images import binary
