import cv2

def run_stereo_sgbm(leftimg, rightimg):
    # Load images
    img_left = cv2.imread(leftimg, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(rightimg, cv2.IMREAD_GRAYSCALE)

    # SGBM parameters
    win_size = 25
    min_disp = 0
    num_disp = 256
    uniqueness_ratio = 10
    speckle_window_size = 100
    speckle_range = 32
    disp_max_diff = 100

    # Create the StereoSGBM object and compute the disparity map
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=win_size,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        disp12MaxDiff=disp_max_diff,
        P1=8*3*win_size**2,
        P2=32*3*win_size**2
    )

    disparity = stereo.compute(img_left, img_right)
    disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity_norm