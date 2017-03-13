import glob
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import numpy as np
from moviepy.editor import VideoFileClip


def display_images(images, cmap=None, col=3, title=None):
    row = (len(images)-1)//col + 1
    gs = gridspec.GridSpec(row, col)
    fig = plt.figure(figsize=(30,10))
    for im,g in zip(images, gs):
        s = fig.add_subplot(g)
        s.imshow(im, cmap=cmap)
        if title: s.set_title(title)
    gs.tight_layout(fig)
    plt.show()

def findChessboardCorners(im, dim):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCorners(gray, dim, None)
    ret = cv2.drawChessboardCorners(np.copy(im), dim, corners, found) if found else im
    return ret, found, corners

def threshold(im, mn, mx):
    im = im.astype(np.float32)
    norm = im / np.max(im)
    fltr = np.logical_and(norm >= mn, norm <= mx)
    thresh = np.zeros_like(im, dtype=np.float32)
    thresh[fltr] = norm[fltr]
    return thresh

def gradients(im, ksize=13):
    gray = np.uint8(im)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    direction = np.arctan2(abs_sobel_y, abs_sobel_x)
    return abs_sobel_x, abs_sobel_y, magnitude, direction

def slide_windows(im, out_img, slide_start, num_window, width_window, thresh=50):
    imageh, imagew = im.shape
    height_window = imageh // num_window
    pos = slide_start
    xpts, ypts = np.int8([]), np.int8([])
    for w in range(num_window):
        windex = num_window - w
        lowx, lowy = pos - width_window // 2, (windex - 1) * height_window
        highx, highy = pos + width_window // 2, windex * height_window
        nzy, nzx = np.nonzero(im[lowy:highy, lowx:highx])
        xpts = np.append(xpts, nzx + lowx)
        ypts = np.append(ypts, nzy + lowy)
        if (len(nzx) > thresh):
            best_index = np.int(np.mean(nzx))
            pos = pos - width_window // 2 + best_index
        cv2.rectangle(out_img, (lowx, lowy), (highx, highy), (0, 255, 0), 2)
    return xpts, ypts


def radius_of_curvature(fit, y):
    a, b, c = fit
    return ((1 + (2 * a * y + b) ** 2) ** (3 / 2)) / np.absolute(2 * a)

def offset_from_centre(image, left, right):
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700

    lxp, rxp = left.xp * xm_per_pix, right.xp * xm_per_pix
    lyp, ryp = left.yp * ym_per_pix, right.yp * ym_per_pix
    real_left, real_right = np.polyfit(lyp, lxp, 2), np.polyfit(ryp, rxp, 2)

    centre_x, centre_y = image.shape[1]//2 * xm_per_pix, image.shape[0] * ym_per_pix
    coeff = np.array([centre_y**2, centre_y, 1])

    lp = np.dot(real_left, coeff.T)
    rp = np.dot(real_right, coeff.T)
    return centre_x - (lp + rp)/2

class Camera:
    def __init__(self, images):
        self.mtx, self.dist = self.calibrate(images)

    def calibrate(self, images):
        r, c = (9, 6)
        processed = np.vstack([findChessboardCorners(im, (r, c)) for im in images])
        valid = processed[processed[:, 1] == True]
        image_shape = valid[0, 0].shape
        image_points = np.array([v.squeeze() for v in valid[:, 2]])
        coords = np.zeros((r * c, 3), np.float32)
        coords[:, :2] = np.mgrid[:r, :c].T.reshape(-1, 2)
        object_points = np.tile(coords, (image_points.shape[0], 1, 1))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_shape[0:2], None,
                                                           None)
        return mtx, dist


class Frame:
    def __init__(self, image):
        self.image = image
        self.binary = None
        self.warped = None
        self.annotated = None
        self.debug = {}

    def add_debug(self, name, value):
        self.debug[name] = value

    def undistort(self, camera):
        undistorted = cv2.undistort(self.image, camera.mtx, camera.dist, None, camera.mtx)
        return Frame(undistorted)

    def channels(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        hls = cv2.cvtColor(self.image, cv2.COLOR_RGB2HLS)
        hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        yuv = cv2.cvtColor(self.image, cv2.COLOR_RGB2YUV)
        gg, r, g, b, h, l, s, hh, ss, vv, y, u, yv = [
            gray, self.image[:,:,0], self.image[:,:,1], self.image[:,:,2],
            hls[:,:,0], hls[:,:,1], hls[:,:,2],
            hsv[:,:,0], hsv[:,:,1], hsv[:,:,2],
            yuv[:,:,0], yuv[:,:,1], yuv[:,:,2]]
        return r, g, b, h, l, s, hh, ss, vv, y, u, yv, gg

    def thresholded_color(self):
        r, g, b, h, l, s, hh, ss, vv, y, u, yv, _ = self.channels()
        #thresholded_r = threshold(r, 0.75, 1.0)
        thresholded_s = threshold(s, 0.75, 1.0)
        thresholded_l = threshold(l, 0.90, 1.0)
        #thresholded_ss = threshold(ss, 0.70, 1.0)
        #thresholded_vv = threshold(vv, 0.90, 1.0)
        #thresholded_y = threshold(y, 0.90, 1.0)
        return thresholded_s, thresholded_l

    def threshold_gradients(self, im):
        x_grad, y_grad, mag_grad, dir_grad = gradients(im)
        x_grad = threshold(x_grad, 0.3, 1.0)
        y_grad = threshold(y_grad, 0.3, 1.0)
        mag_grad = threshold(mag_grad, 0.3, 1.0)
        dir_grad = threshold(dir_grad, 0.5, 0.8)
        return x_grad, y_grad, mag_grad, dir_grad

    def threshold_binary(self):
        thresh_s, thresh_l = self.thresholded_color()
        color_combined = np.logical_or(thresh_s, thresh_l)
        x_grad, y_grad, mag_grad, dir_grad = self.threshold_gradients(color_combined)
        grad_combined = np.logical_and(x_grad, y_grad)
        dir_combined = np.logical_and(dir_grad, mag_grad)
        combined = np.logical_or(grad_combined, dir_combined)
        return Frame(combined)

    def warp_image(self):
        # src = np.float32([[190,720],[580,450],[700,450],[1170,720]])
        src = np.float32([[200, 720], [590, 450], [690, 450], [1100, 720]])
        dst = np.float32([[200, 720], [200, 0], [1080, 0], [1080, 720]])
        maxy, maxx = self.image.shape[0], self.image.shape[1]
        M = cv2.getPerspectiveTransform(src, dst)
        unsigned = np.uint8(self.image)
        warped = cv2.warpPerspective(unsigned, M, (maxx, maxy), flags=cv2.INTER_LINEAR)
        return Warped(warped, src, dst)

    def apply_mask(self, mask, ratio):
        masked = cv2.addWeighted(self.image, 1, mask, ratio, 0)
        return Frame(masked)

    def birds_eye_view(self, camera):
        undistorted = self.undistort(camera)
        thresholded = undistorted.threshold_binary()
        warped = thresholded.warp_image()
        return warped

    def mark_lanes(self, mask, left, right, ratio=0.3):
        average_fit = np.average((left.fit, right.fit), axis=0, weights=(len(left.xp), len(right.xp)))
        rc = radius_of_curvature(average_fit, mask.shape[0])
        oc = offset_from_centre(self.image, left, right)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(mask, "Radius of curvature : " + str(rc) + " m", (100, 50), font, 1.5, (255, 255, 255), 2)
        cv2.putText(mask, "Offset from centre  : " + str(oc) + " m", (100, 100), font, 1.5, (255, 255, 255), 2)
        return self.apply_mask(mask, ratio)

class Warped(Frame):
    def __init__(self, image, src, dst):
        Frame.__init__(self, image)
        self.src = src
        self.dst = dst

    def color_image(self):
        return np.dstack((self.image, self.image, self.image)) * 255

    def unwarp(self):
        maxy, maxx = self.image.shape[0], self.image.shape[1]
        unsigned = np.uint8(self.image)
        Minv = cv2.getPerspectiveTransform(self.dst, self.src)
        unwarped = cv2.warpPerspective(unsigned, Minv, (maxx, maxy), flags=cv2.INTER_LINEAR)
        return Frame(unwarped)

    def search_for_fit(self):
        warped = self.image
        h, w = warped.shape[0], warped.shape[1]
        top, bottom = warped[:h // 2, :], warped[h // 2:, :]
        hist_bottom = np.sum(bottom, axis=0)
        left_bottom, right_bottom = np.argmax(hist_bottom[:w // 2]), w // 2 + np.argmax(hist_bottom[w // 2:])

        out_img = np.dstack((warped, warped, warped)) * 255
        lxp, lyp = slide_windows(warped, out_img, left_bottom, 9, 100)
        rxp, ryp = slide_windows(warped, out_img, right_bottom, 9, 100)
        out_img[lyp, lxp] = [255, 0, 0]
        out_img[ryp, rxp] = [0, 0, 255]

        self.add_debug("sliding_window_image", out_img)

        left_lane = Line(lxp, lyp, left_bottom) if (len(lxp) > 0) else None
        right_lane = Line(rxp, ryp, right_bottom) if (len(rxp) > 0) else None

        return left_lane, right_lane

    def search_around_fit(self, fit, margin=50):
        im = self.image
        lfit, rfit = fit.left_fit, fit.right_fit
        nzy, nzx = np.nonzero(im)
        coeff = np.array([[y ** 2, y, 1] for y in nzy])
        plx = np.dot(lfit, coeff.T)
        prx = np.dot(rfit, coeff.T)
        lcond = (nzx - margin < plx) & (nzx + margin > plx)
        rcond = (nzx - margin < prx) & (nzx + margin > prx)
        lx, ly, rx, ry = nzx[lcond], nzy[lcond], nzx[rcond], nzy[rcond]

        return Fit(self, lx, ly, rx, ry, [], [])

    def lane_mask(self, left, right):
        im = np.dstack((self.image, self.image, self.image))
        h, w = im.shape[0], im.shape[1]
        y_values = np.arange(h)
        coeff = np.array([[y ** 2, y, 1] for y in y_values])
        lx_values = np.dot(left.fit, coeff.T)
        rx_values = np.dot(right.fit, coeff.T)

        color_warp = np.zeros_like(im).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([lx_values, y_values]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rx_values, y_values])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        return Warped(color_warp, self.src, self.dst).unwarp()

class Line:
    def __init__(self, xp, yp, bottom):
        self.xp = xp
        self.yp = yp
        self.bottom = bottom
        self.fit = np.polyfit(self.yp, self.xp, 2)

    def draw_on(self, image):
        h, w = image.shape[0], image.shape[1]
        y_values = np.arange(h)
        coeff = np.array([[y ** 2, y, 1] for y in y_values])
        x_values = np.dot(self.fit, coeff.T)

        window_img = np.zeros_like(image)
        points1 = np.vstack((x_values - 50, y_values)).T.astype(np.int32)
        points2 = np.vstack((x_values + 50, y_values)).T.astype(np.int32)
        points = np.vstack((points1, np.flipud(points2)))
        cv2.fillPoly(window_img, [points], (0, 255, 0))
        result = cv2.addWeighted(image, 1, window_img, 0.3, 0)

        for x,y in zip(x_values, y_values): cv2.circle(result, (int(x),int(y)), 5, (255,0,0), 10)
        return result


class AlteredLine(Line):
    def __init__(self, fit, bottom):
        self.fit = fit
        self.bottom = bottom
        self.yp = []
        self.xp = []

class LaneFinder:
    def __init__(self, camera):
        #self.fits = []
        self.camera = camera
        self.left_fit = np.array([])
        self.right_fit = np.array([])
        self.best_left = None
        self.best_right = None

    def validate_fit(self, lfit, rfit, lx, ly, rx, ry, bottom_width):
        mx, mn = np.max((len(lx), len(rx))), np.min((len(lx), len(rx)))
        if (mx < 8000 and self.left_fit.any() and self.right_fit.any()):
            lfit, rfit = self.left_fit, self.right_fit
            lx, ly, rx, ry = self.lx, self.ly, self.rx, self.ry
        elif (mn < mx // 2):
            if len(lx) < len(rx):
                lx = (rx - bottom_width)
                ly = ry
                lfit = np.polyfit(ly, lx, 2)
            else:
                rx = (lx + bottom_width)
                ry = ly
                rfit = np.polyfit(ry, rx, 2)
        return lfit, rfit, lx, ly, rx, ry

    def find_lane(self, im):
        frame = Frame(im)
        warped = frame.birds_eye_view(self.camera)
        left, right = warped.search_for_fit()
        if not left: left = self.best_left
        if not right: right = self.best_right
        self.best_left = left
        self.best_right = right

        lfit, rfit, lx, ly, rx, ry = left.fit, right.fit, left.xp, left.yp, right.xp, right.yp
        bottom_width = right.bottom - left.bottom
        lfit, rfit, lx, ly, rx, ry = self.validate_fit(lfit, rfit, lx, ly, rx, ry, bottom_width)
        self.left_fit, self.right_fit = lfit, rfit
        self.lx, self.ly, self.rx, self.ry = lx, ly, rx, ry

        mask = warped.lane_mask(AlteredLine(lfit, left.bottom), AlteredLine(rfit, right.bottom))
        ann = frame.mark_lanes(mask.image, left, right)

        return ann.image


calibration_images = [mpimg.imread(f) for f in glob.glob(os.path.join("camera_cal", "calibration*.jpg"))]

# # Calibrations
camera = Camera(calibration_images)
# sample_image = calibration_images[0]
# frame = Frame(sample_image)
# frame.undistort(camera)
# display_images([sample_image, frame.image])

# # Channel extraction
# test_dir = "test_images"
# straight_lines1 = mpimg.imread(os.path.join(test_dir, "straight_lines1.jpg"))
# straight_lines2 = mpimg.imread(os.path.join(test_dir, "straight_lines2.jpg"))
# frame = Frame(straight_lines1)
# channels = frame.channels()
# display_images(channels, cmap='gray')

test_images = [mpimg.imread(f) for f in glob.glob(os.path.join("test_images", "test*.jpg"))]
test_frames = [Frame(im) for im in test_images]

# # Choosing channels
# display_images(test_images, title="Test images")
# channels = [ch for im in test_images for ch in Frame(im).channels()[4:6] ]
# display_images(channels, title="L & S channels", col=4, cmap='gray')

# Pipeline
final = []
for frame in test_frames:
    warped = frame.birds_eye_view(camera)
    left, right = warped.search_for_fit()

    im = warped.debug["sliding_window_image"]

    # im = warped.color_image()
    # im = left.draw_on(im)
    # im = right.draw_on(im)

    # mask = warped.lane_mask(left, right)
    # ann = frame.mark_lanes(mask.image, left, right)

    final.append(im)

display_images(final, title="Final")

# # Video
# video_output = 'see.mp4'
# clip1 = VideoFileClip("project_video.mp4")
# annotated = clip1.fl_image(LaneFinder(camera).find_lane)
# annotated.write_videofile(video_output, audio=False)

# # Debug
# image = mpimg.imread(os.path.join("exported", "frame0612.jpeg"))
# thresh = Frame(image).undistort(camera).threshold_binary()
# warp = thresh.warp_image()
# display_images([image, thresh.image, warp.image])
