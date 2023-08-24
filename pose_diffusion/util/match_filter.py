import numpy as np
import cv2



def cross_check_matches(descriptors_a, descriptors_b):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches_ab = matcher.match(descriptors_a, descriptors_b)
    matches_ba = matcher.match(descriptors_b, descriptors_a)

    cross_checked_matches = []
    for match_ab in matches_ab:
        match_ba = matches_ba[match_ab.trainIdx]
        if match_ab.queryIdx == match_ba.trainIdx:
            cross_checked_matches.append((match_ab.queryIdx, match_ab.trainIdx))

    return cross_checked_matches

def export_colmap_matches(image_id1, image_id2, cross_checked_matches, output_file):
    with open(output_file, 'w') as f:
        f.write(f"{image_id1} {image_id2}\n")
        f.write(f"{len(cross_checked_matches)}\n")
        for query_idx, train_idx in cross_checked_matches:
            f.write(f"{query_idx} {train_idx}\n")

def contrast_stretching(image, min_val=0, max_val=255):
    in_min, in_max = np.min(image), np.max(image)
    stretched_image = (image - in_min) * ((max_val - min_val) / (in_max - in_min)) + min_val
    return stretched_image.astype(np.uint8)


def histogram_equalization(image, is_color=True):
    if is_color:
        # Convert to YCrCb color space
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        # Perform histogram equalization on the Y channel
        ycrcb_image[..., 0] = cv2.equalizeHist(ycrcb_image[..., 0])
        # Convert back to BGR color space
        equalized_image = cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2BGR)
    else:
        equalized_image = cv2.equalizeHist(image)
    return equalized_image




def filter_matches(keypoints_a, keypoints_b, matches, method='fundamental', threshold=1.0, confidence=0.99, max_iters=1000):
    if len(matches) < 8:
        return []

    src_pts = np.float32([keypoints_a[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_b[m.trainIdx] for m in matches]).reshape(-1, 1, 2)

    if method == 'fundamental':
        # Compute the fundamental matrix
        M, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, threshold, confidence, max_iters)
    elif method == 'essential':
        # Compute the essential matrix, assuming you have camera intrinsics matrix K
        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, threshold, confidence, max_iters)
        M = np.matmul(np.matmul(K.T, F), K)
    elif method == 'homography':
        # Compute the homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold, confidence=confidence, maxIters=max_iters)
    else:
        raise ValueError("Invalid method: choose either 'fundamental', 'essential', or 'homography'")

    # Filter matches based on the mask
    filtered_matches = [m for m, m_valid in zip(matches, mask.ravel()) if m_valid]

    return filtered_matches