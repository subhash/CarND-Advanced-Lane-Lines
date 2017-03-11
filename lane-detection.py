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

    def find_fit(self, camera):
        undistorted = self.undistort(camera)
        thresholded = undistorted.threshold_binary()
        warped = thresholded.warp_image()
        fit = warped.search_for_fit()
        return fit

    def mark_lane(self, fit, ratio=0.3):
        mask = fit.mask().unwarp().image
        average_fit = fit.to_real_world().average()
        rc = radius_of_curvature(average_fit, mask.shape[0])
        oc = fit.to_real_world().offset_from_centre()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(mask, "Radius of curvature : " + str(rc) + " m", (100, 50), font, 1.5, (255, 255, 255), 2)
        cv2.putText(mask, "Offset from centre  : " + str(oc) + " m", (100, 100), font, 1.5, (255, 255, 255), 2)
        return self.apply_mask(mask, ratio)

class Warped(Frame):
    def __init__(self, image, src, dst):
        Frame.__init__(self, image)
        self.src = src
        self.dst = dst

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

        if (len(lxp) > 0 and len(rxp) > 0):
            fit =  Fit(self, lxp, lyp, rxp, ryp, left_bottom, right_bottom)
            fit.add_debug("sliding_window_image", out_img)
            return fit
        else:
            return None

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


class Fit(Warped):

    def __init__(self, warped, lxp, lyp, rxp, ryp, left_bottom, right_bottom):
        Warped.__init__(self, warped.image, warped.src, warped.dst)
        self.lxp = lxp
        self.lyp = lyp
        self.rxp = rxp
        self.ryp = ryp
        self.left_fit = np.polyfit(self.lyp, self.lxp, 2)
        self.right_fit = np.polyfit(self.ryp, self.rxp, 2)
        self.left_bottom = left_bottom
        self.right_bottom = right_bottom

    def to_real_world(self):
        ym_per_pix = 30 / 720
        xm_per_pix = 3.7 / 700
        lxp, rxp = self.lxp * xm_per_pix, self.rxp * xm_per_pix
        lyp, ryp = self.lyp * ym_per_pix, self.ryp * ym_per_pix
        return Fit(self, lxp, lyp, rxp, ryp, [], [])

    def offset_from_centre(self):
        ym_per_pix = 30 / 720
        xm_per_pix = 3.7 / 700
        centre_x, centre_y = self.image.shape[1]//2 * xm_per_pix, self.image.shape[0] * ym_per_pix
        coeff = np.array([centre_y**2, centre_y, 1])
        lp = np.dot(self.left_fit, coeff.T)
        rp = np.dot(self.right_fit, coeff.T)
        return centre_x - (lp + rp)/2

    def average(self):
        ave = np.average((self.left_fit, self.right_fit), axis=0, weights=(len(self.lxp), len(self.rxp)))
        return ave

    def mask(self):
        im = np.dstack((self.image, self.image, self.image))
        h, w = im.shape[0], im.shape[1]
        y_values = np.arange(h)
        coeff = np.array([[y ** 2, y, 1] for y in y_values])
        lx_values = np.dot(self.left_fit, coeff.T)
        rx_values = np.dot(self.right_fit, coeff.T)

        color_warp = np.zeros_like(im).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([lx_values, y_values]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([rx_values, y_values])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        return Warped(color_warp, self.src, self.dst)


    def plot(self):
        out_img = np.dstack((self.image, self.image, self.image)) * 255
        h, w = out_img.shape[0], out_img.shape[1]
        y_values = np.arange(h)
        coeff = np.array([[y ** 2, y, 1] for y in y_values])
        lx_values = np.dot(self.left_fit, coeff.T)
        rx_values = np.dot(self.right_fit, coeff.T)

        window_img = np.zeros_like(out_img)
        l1points = np.vstack((lx_values - 50, y_values)).T.astype(np.int32)
        l2points = np.vstack((lx_values + 50, y_values)).T.astype(np.int32)
        r1points = np.vstack((rx_values - 50, y_values)).T.astype(np.int32)
        r2points = np.vstack((rx_values + 50, y_values)).T.astype(np.int32)
        lpoints = np.vstack((l1points, np.flipud(l2points)))
        rpoints = np.vstack((r1points, np.flipud(r2points)))
        cv2.fillPoly(window_img, [lpoints], (0, 255, 0))
        cv2.fillPoly(window_img, [rpoints], (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        for x,y in zip(lx_values, y_values): cv2.circle(result, (int(x),int(y)), 5, (255,0,0), 10)
        for x,y in zip(rx_values, y_values): cv2.circle(result, (int(x),int(y)), 5, (0,0,255), 10)

        return result

class AlteredFit(Fit):
    def __init__(self, fit, left_fit, right_fit):
        Fit.__init__(self, fit, fit.lxp, fit.lyp, fit.rxp, fit.ryp, fit.left_bottom, fit.right_bottom)
        self.left_fit = left_fit
        self.right_fit = right_fit


class LaneFinder:
    def __init__(self, camera):
        #self.fits = []
        self.camera = camera
        self.left_fit = np.array([])
        self.right_fit = np.array([])
        self.best_fit = None

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
        fit = frame.find_fit(self.camera)
        if not fit:
            fit = self.best_fit
        else:
            self.best_fit = fit
        lfit, rfit, lx, ly, rx, ry = fit.left_fit, fit.right_fit, fit.lxp, fit.lyp, fit.rxp, fit.ryp
        bottom_width = fit.right_bottom - fit.left_bottom
        lfit, rfit, lx, ly, rx, ry = self.validate_fit(lfit, rfit, lx, ly, rx, ry, bottom_width)
        self.left_fit, self.right_fit = lfit, rfit
        self.lx, self.ly, self.rx, self.ry = lx, ly, rx, ry
        marked = frame.mark_lane(AlteredFit(fit, lfit, rfit))
        return marked.image


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

# test_images = [mpimg.imread(f) for f in glob.glob(os.path.join("test_images", "test*.jpg"))]
# test_frames = [Frame(im) for im in test_images]

# # Choosing channels
# display_images(test_images, title="Test images")
# channels = [ch for im in test_images for ch in Frame(im).channels()[4:6] ]
# display_images(channels, title="L & S channels", col=4, cmap='gray')

# Pipeline
# final = [frame.mark_lane(frame.find_fit(camera)).image for frame in test_frames]
# display_images(final, title="Final")

# Video
video_output = 'see.mp4'
clip1 = VideoFileClip("project_video.mp4")
annotated = clip1.fl_image(LaneFinder(camera).find_lane)
annotated.write_videofile(video_output, audio=False)

# # Debug
# image = mpimg.imread(os.path.join("exported", "frame0612.jpeg"))
# thresh = Frame(image).undistort(camera).threshold_binary()
# warp = thresh.warp_image()
# display_images([image, thresh.image, warp.image])
