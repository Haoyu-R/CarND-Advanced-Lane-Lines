import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob


def camera_calibration(path, rows, columns):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(path)

    image_shape = None
    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        if image_shape is None:
            image_shape = img.shape[1::-1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (columns, rows), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

    return mtx, dist


def binary_thresholding(img, s_threshold, lx_threshold):
    # bgr to hls since using glob to import img
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls_img[:, :, 2]
    l_channel = hls_img[:, :, 1]
    # Sobel x on L channel
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    lxbinary = np.zeros_like(scaled_sobelx)
    lxbinary[(scaled_sobelx > lx_threshold[0]) & (scaled_sobelx < lx_threshold[1])] = 1

    sbinary = np.zeros_like(scaled_sobelx)
    sbinary[(s_channel > s_threshold[0]) & (s_channel < s_threshold[1])] = 1

    result = mask_vertices(lxbinary + sbinary)

    return result


def new_line_search(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 75
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin if (leftx_current - margin) > 0 else 0  # Update this
        win_xleft_high = leftx_current + margin if (leftx_current + margin) < binary_warped.shape[1] else \
            binary_warped.shape[1]
        win_xright_low = rightx_current - margin if (rightx_current - margin) > 0 else 0  # Update this
        win_xright_high = rightx_current + margin if (rightx_current + margin) < binary_warped.shape[1] else \
            binary_warped.shape[1]

        # Draw the windows on the visualization image
        # out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low),
        #               (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low),
        #               (win_xright_high, win_y_high), (0, 255, 0), 2)

        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = np.intersect1d(np.argwhere((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)),
                                        np.argwhere((nonzeroy < win_y_high) & (nonzeroy >= win_y_low)))
        good_right_inds = np.intersect1d(np.argwhere((nonzerox >= win_xright_low) & (nonzerox < win_xright_high)),
                                         np.argwhere((nonzeroy < win_y_high) & (nonzeroy >= win_y_low)))

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        ### If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        # If the window is approching the top of image, reduce the minimum required pixel in the consideration of curve
        if window > 4:
            if good_left_inds.size > (minpix - window * 3):
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if good_right_inds.size > (minpix - window * 3):
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        else:
            if good_left_inds.size > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if good_right_inds.size > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        # try:
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # except ValueError:
    #     # Avoids an error if the above is not implemented fully
    #     pass
    if left_lane_inds.size == 0 or right_lane_inds.size == 0:
        return (0, 0, 0), (0, 0, 0)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # # avg0 = (left_fit[0] + right_fit[0])/2
    # avg1 = (left_fit[1] + right_fit[1])/2
    # # left_fit[0] = avg0
    # # right_fit[0] = avg0
    # left_fit[1] = avg1
    # right_fit[1] = avg1

    return left_fit, right_fit


def line_search(binary_warped, left_fit, right_fit):
    margin = 75

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###

    left_lane_inds = (((nonzerox - margin) < (left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2])) &
                      ((nonzerox + margin) > (left_fit[0] * nonzeroy ** 2 + left_fit[1] * nonzeroy + left_fit[2])))
    right_lane_inds = (((nonzerox - margin) < (right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2])) &
                       ((nonzerox + margin) > (right_fit[0] * nonzeroy ** 2 + right_fit[1] * nonzeroy + right_fit[2])))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if np.sum(left_lane_inds) == 0 or np.sum(right_lane_inds) == 0:
        return (0, 0, 0), (0, 0, 0)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


def draw_line_area(color_warp, left_fit, right_fit):
    # Generate x and y values for plotting

    ploty = np.linspace(0, color_warp.shape[0] - 1, color_warp.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    left_negative = np.argwhere(left_fitx < 0)
    right_positive = np.argwhere(right_fitx > color_warp.shape[1] - 23)
    if len(left_negative) > 25:
        idx = np.max(left_negative)
        left_fitx = left_fitx[idx:]
        right_fitx = right_fitx[idx:]
        ploty = ploty[idx:]
    if len(right_positive) > 25:
        idx = np.max(right_positive)
        left_fitx = left_fitx[idx:]
        right_fitx = right_fitx[idx:]
        ploty = ploty[idx:]
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    if pts.size == 0:
        return color_warp
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    return color_warp


def measure_curvature_real(y_max, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 50 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 900  # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = y_max * ym_per_pix

    ##### Implement the calculation of R_curve (radius of curvature) #####
    left_curverad = (1 + (2 * left_fit[0] * y_eval * xm_per_pix / (ym_per_pix ** 2) + left_fit[
        1] * xm_per_pix / ym_per_pix) ** 2) ** 1.5 / abs(
        2 * left_fit[0] * xm_per_pix / ym_per_pix ** 2)
    right_curverad = (1 + (2 * right_fit[0] * y_eval * xm_per_pix / ym_per_pix ** 2 + right_fit[
        1] * xm_per_pix / ym_per_pix) ** 2) ** 1.5 / abs(
        2 * right_fit[0] * xm_per_pix / ym_per_pix ** 2)
    return left_curverad,right_curverad


def measure_center_diviation(x_max, y_max, left_fit, right_fit):
    xm_per_pix = 3.7 / 900
    x_left_bottom = left_fit[0] * y_max ** 2 + left_fit[1] * y_max + left_fit[2]
    x_right_bottom = right_fit[0] * y_max ** 2 + right_fit[1] * y_max + right_fit[2]

    return (x_max - (x_left_bottom+x_right_bottom))/2 * xm_per_pix


def write_text(img, curvature, deviation, left_fit, right_fit):
    # Write some Text
    string1 = "Radius of Curvature = {:.1f}(m)".format(curvature)
    string2 = ""
    # string3 = "A: {:.8f}, {:.8f}".format(left_fit[0], right_fit[0])
    # string4 = "B: {:.8f}, {:.8f}".format(left_fit[1], right_fit[1])

    if deviation < 0:
        string2 = "Vehicle is {:.2f}m left of center".format(abs(deviation))
    else:
        string2 = "Vehicle is {:.2f}m right of center".format(deviation)

    bottomLeftCornerOfText1 = (10, 50)
    bottomLeftCornerOfText2 = (10, 100)
    # bottomLeftCornerOfText3 = (10, 150)
    # bottomLeftCornerOfText4 = (10, 200)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img, string1,
                bottomLeftCornerOfText1,
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.putText(img, string2,
                bottomLeftCornerOfText2,
                font,
                fontScale,
                fontColor,
                lineType)

    # cv2.putText(img, string3,
    #             bottomLeftCornerOfText3,
    #             font,
    #             fontScale,
    #             fontColor,
    #             lineType)
    # cv2.putText(img, string4,
    #             bottomLeftCornerOfText4,
    #             font,
    #             fontScale,
    #             fontColor,
    #             lineType)
    return img


def mask_vertices(img):
    # Define region of interest
    vertices = np.array([[0, 719], [598, 440], [682, 440], [1279, 719]])
    conjunction_img = np.zeros_like(img)
    cv2.fillPoly(conjunction_img, [vertices], 1)
    masked = cv2.bitwise_and(img, conjunction_img)
    return masked


def mask_color(img):
    # Define color of interest
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 155], dtype=np.uint8)
    upper_white = np.array([255, 100, 255], dtype=np.uint8)
    lower_blue = np.array([110, 100, 100], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([40, 255, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv_img, lower_white, upper_white)
    mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    masked_color = mask_white + mask_blue + mask_yellow

    return masked_color


def measure_top_distance(left_fit, right_fit):
    xm_per_pix = 3.7 / 900
    return abs(left_fit[2] - right_fit[2]) * xm_per_pix


def lane_detection_pipline(img, mtx, dist, M, invM, line):
    # some hyperparameters
    s_threshold = (170, 255)
    lx_threshold = (20, 100)

    # undistort img
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    thresh_binary_img = binary_thresholding(undist, s_threshold, lx_threshold)

    img_size = (img.shape[1], img.shape[0])
    warped_img = cv2.warpPerspective(thresh_binary_img, M, img_size, flags=cv2.INTER_LINEAR)

    if not line.detected:
        left_fit, right_fit = new_line_search(warped_img)
        if np.sum(left_fit) == 0 or np.sum(right_fit) == 0:
            return undist
        line.insert(left_fit, right_fit)
    else:
        left_fit, right_fit = line_search(warped_img, line.last_left, line.last_right)
        if np.sum(left_fit) == 0 or np.sum(right_fit) == 0:
            left_fit, right_fit = new_line_search(warped_img)
            if np.sum(left_fit) == 0 or np.sum(right_fit) == 0:
                line.reset()
                return undist
        line.insert(left_fit, right_fit)

    left_fit, right_fit = line.get_coefficient()
    left_curvature, right_curvature = measure_curvature_real(img.shape[0], left_fit, right_fit)
    deviation = measure_center_diviation(img.shape[1], img.shape[0], left_fit, right_fit)

    if abs(deviation) > 1 or measure_top_distance(left_fit, right_fit) < 2.5 or abs(left_fit[1] - right_fit[1]) > 2.5:
        if line.detected:
            line.pop_last()
            left_fit, right_fit = line.get_coefficient()
            left_curvature, right_curvature = measure_curvature_real(img.shape[0], left_fit, right_fit)
            deviation = measure_center_diviation(img.shape[1], img.shape[0], left_fit, right_fit)
            if abs(deviation) > 1 or measure_top_distance(left_fit, right_fit) < 2.5 or abs(
                left_fit[1] - right_fit[1]) > 2.5:
                left_fit, right_fit = new_line_search(warped_img)
                if np.sum(left_fit) == 0 or np.sum(right_fit) == 0:
                    line.reset()
                    return undist
                line.insert(left_fit, right_fit)
                left_fit, right_fit = line.get_coefficient()
                left_curvature, right_curvature = measure_curvature_real(img.shape[0], left_fit, right_fit)
                deviation = measure_center_diviation(img.shape[1], img.shape[0], left_fit, right_fit)

    line.detected = True

    # print(left_curvature, right_curvature, deviation)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    line_warp = draw_line_area(color_warp, left_fit, right_fit)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(line_warp, invM, img_size)
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    result = write_text(result, (left_curvature+right_curvature)/2, deviation, left_fit, right_fit)

    resize_warped = cv2.resize(line_warp, (160, 90), interpolation=cv2.INTER_AREA)
    # rgb_warped = cv2.cvtColor(resize_warped, cv2.COLOR_GRAY2BGR)

    result[0:90, result.shape[1]- 160: result.shape[1], :] = resize_warped

    # plt.figure()
    # f, ax = plt.subplots(1, 2)
    # ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # ax[0].imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
    # ax[0].imshow(thresh_binary_img)
    # ax[0].imshow(warped_img)
    # ax[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    # plt.show()

    return result
