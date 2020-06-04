# ## Advanced Lane Finding Project
#
# The goals / steps of this project are the following:
#
# * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# * Apply a distortion correction to raw images.
# * Use color transforms, gradients, etc., to create a thresholded binary image.
# * Apply a perspective transform to rectify binary image ("birds-eye view").
# * Detect lane pixels and fit to find the lane boundary.
# * Determine the curvature of the lane and vehicle position with respect to center.
# * Warp the detected lane boundaries back onto the original image.
# * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
#


from examples.process_utils import *
from examples.class_line import *
from moviepy.editor import VideoFileClip

# Calibrate the camera
calibration_path = r'../camera_cal/calibration*.jpg'
mtx, dist = camera_calibration(calibration_path, 6, 9)

# Pre-define points for perspective transformation
# scr_points = np.array([[50, 719], [580, 450], [705, 450], [1279, 719]], np.float32)
scr_points = np.array([[0, 719], [598, 440], [682, 440], [1279, 719]], np.float32)
dst_points = np.array([[0, 719], [0, 0], [1279, 0], [1279, 719]], np.float32)
M = cv2.getPerspectiveTransform(scr_points, dst_points)
invM = cv2.getPerspectiveTransform(dst_points, scr_points)

# cal_path = calibration_path = r'../camera_cal/calibration4.jpg'
# cal_img = cv2.imread(cal_path)
# undist = cv2.undistort(cal_img, mtx, dist, None, mtx)
# plt.figure()
# f, ax = plt.subplots(1, 2)
# ax[0].imshow(cv2.cvtColor(cal_img, cv2.COLOR_BGR2RGB))
# ax[1].imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))


def video_process(img):
    processed_img = lane_detection_pipline(img, mtx, dist, M, invM, line)
    return processed_img


# Pipline software
# images = glob.glob(r'../test_images/*.jpg')
#
# for image in images:
#     line = Line()
#     img = cv2.imread(image)
#     processed_img = lane_detection_pipline(img, mtx, dist, M, invM, line)
#     processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    # plt.imshow(processed_img)
    # plt.show()

line = Line()
clip = VideoFileClip(r'../project_video.mp4')
new_clip = clip.fl_image(video_process)
new_clip.write_videofile(r'../project_video_test.mp4', audio=False)