import glob
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import numpy as np

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

    def apply_mask(self, mask, ratio=0.3):
        masked = cv2.addWeighted(self.image, 1, mask, ratio, 0)
        return Frame(masked)

    def mark_lane(self, camera):
        undistorted = self.undistort(camera)
        thresholded = undistorted.threshold_binary()
        warped = thresholded.warp_image()
        fit = warped.search_for_fit()
        mask = fit.mask().unwarp().image
        average_fit = fit.to_real_world().average()
        rc = radius_of_curvature(average_fit, mask.shape[0])
        oc = fit.to_real_world().offset_from_centre()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(mask, "Radius of curvature : " + str(rc) + " m", (100, 50), font, 1.5, (255, 255, 255), 2)
        cv2.putText(mask, "Offset from centre  : " + str(oc) + " m", (100, 100), font, 1.5, (255, 255, 255), 2)
        return self.apply_mask(mask)

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

        fit =  Fit(self, lxp, lyp, rxp, ryp, left_bottom, right_bottom)
        fit.add_debug("sliding_window_image", out_img)
        return fit

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
final = [frame.mark_lane(camera).image for frame in test_frames]
display_images(final, title="Final")


# # thresh_images, thresh_s, thresh_color, combined_grad = zip(*[threshold_pipeline(im) for im in test_images])
# # display_images(thresh_color, col=3, cmap='gray', title='Combined Channels')
# # # diff = [c - s for c, s in zip(thresh_color, thresh_s)]
# # # display_images(diff, col=3, cmap='gray', title='Diff Channels')
# # display_images(combined_grad, col=3, cmap='gray', title="Combined gradient")
# # display_images(thresh_images, col=3, cmap='gray', title="Thresholded")
#
# thresh_images = [threshold_pipeline(im) for im in test_images]
# # display_images(test_images, col=3)
# display_images(thresh_images, cmap='gray', col=3)
#
#
# def mask_image(im):
#     im = np.uint8(im)
#     mask = np.zeros_like(im)
#     vertices = np.array([[[180, 720], [570, 450], [710, 450], [1190, 720]]], dtype=np.int32)
#     cv2.fillPoly(mask, vertices, 255)
#     return cv2.bitwise_and(mask, im)
#
#
#
#
#
#
# def plot_warped(im, warped, src, dst):
#     plt.imshow(im, cmap='gray')
#     plt.title("Transform")
#     plt.plot(*zip(src[0], src[1]), 'r-')
#     plt.plot(*zip(src[2], src[3]), 'r-')
#     plt.figure()
#     plt.imshow(warped, cmap='gray')
#     plt.title("Warped")
#     plt.plot(*zip(dst[0], dst[1]), 'r-')
#     plt.plot(*zip(dst[2], dst[3]), 'r-')
#     plt.show()
#
#
#
#
#
# def real_fit(x, y):
#     ym_per_pix = 30 / 720
#     xm_per_pix = 3.7 / 700
#     fit = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
#     return fit
#
#
#
#
# # im = thresh_images[2]
# im = threshold_pipeline(export_images[8])
#
# warped, src, dst = warp_image(im)
# plot_warped(im, warped, src, dst)
#
# out_img = np.dstack((warped, warped, warped)) * 255
# lfit, rfit, lx, ly, rx, ry, _, _ = search_for_fit(warped, out_img)
# real_lfit = real_fit(lx, ly)
# real_rfit = real_fit(rx, ry)
# plt.imshow(out_img)
# plt.title("Sliding windows")
# plt.show()
#
# out_img = np.dstack((warped, warped, warped)) * 255
# plot_fit(out_img, lfit, rfit, title='Sliding window fit')
# nlfit, nrfit, nlx, nly, nrx, nry = search_around_fit(warped, lfit, rfit)
# plot_fit(out_img, nlfit, nrfit, title='Smoother fit')
#
# rl, rr = radius_of_curvature(real_lfit, im.shape[0]), radius_of_curvature(real_rfit, im.shape[0])
# print(rl, 'm ', rr, 'm ')
#
# print(lfit, rfit)
# print(nlfit, nrfit)
# print(len(lx), len(ly), len(rx), len(ry))
# print(len(nlx), len(nly), len(nrx), len(nry))
#
#
# def offset_from_centre(x, y, real_lfit, real_rfit):
#     ym_per_pix = 30 / 720
#     xm_per_pix = 3.7 / 700
#     xm, ym = x * xm_per_pix, y * ym_per_pix
#     coeff = [ym ** 2, ym, 1]
#     lx = np.dot(real_lfit, coeff)
#     rx = np.dot(real_rfit, coeff)
#     return xm - (rx + lx) / 2
#
#
#
#
# def annotate_image(im):
#     thresh = threshold_pipeline(im)
#     warped, src, dst = warp_image(thresh)
#     out_img = np.dstack((warped, warped, warped)) * 255
#     lfit, rfit, lx, ly, rx, ry, _, _ = search_for_fit(warped, out_img)
#     # lfit, rfit, lx, ly, rx, ry = search_around_fit(warped, lfit, rfit)
#
#     mask = lane_mask(im, lfit, rfit)
#
#     # Combine the result with the original image
#     result = cv2.addWeighted(im, 1, mask, 0.3, 0)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     h, w, _ = im.shape
#
#     real_lfit, real_rfit = real_fit(lx, ly), real_fit(rx, ry)
#     rcl, rcr = radius_of_curvature(real_lfit, h), radius_of_curvature(real_rfit, h)
#     off = offset_from_centre(w / 2, h, real_lfit, real_rfit)
#     conf = str(len(lx)) + " - " + str(len(rx))
#     cv2.putText(result, "Radius (Left)      : " + str(rcl), (100, 50), font, 1.5, (255, 255, 255), 2)
#     cv2.putText(result, "Radius (Right)     : " + str(rcr), (100, 100), font, 1.5, (255, 255, 255), 2)
#     cv2.putText(result, "Offset from centre : " + str(off), (100, 150), font, 1.5, (255, 255, 255), 2)
#     cv2.putText(result, "Confidence         : " + str(conf), (100, 200), font, 1.5, (255, 255, 255), 2)
#
#     return result
#
#
# class Annotator:
#     def __init__(self):
#         self.left_fit = np.array([])
#         self.right_fit = np.array([])
#         self.lx = []
#         self.ly = []
#         self.rx = []
#         self.ry = []
#         self.olx = []
#         self.oly = []
#         self.orx = []
#         self.ory = []
#
#     def average_fit(self, lfit, rfit, lx, ly, rx, ry):
#         a, b, _ = np.average((lfit, rfit), axis=0, weights=(len(lx), len(rx)))
#         nlfit = np.array([a, b, lfit[-1]])
#         nrfit = np.array([a, b, rfit[-1]])
#         lcoeff = np.array([[y ** 2, y, 1] for y in ly])
#         rcoeff = np.array([[y ** 2, y, 1] for y in ry])
#         nlx = np.dot(nlfit, lcoeff.T)
#         nrx = np.dot(nrfit, rcoeff.T)
#         return nlfit, nrfit, nlx, ly, nrx, ry
#
#     def validate_fit(self, lfit, rfit, lx, ly, rx, ry, bottom_width):
#         mx, mn = np.max((len(lx), len(rx))), np.min((len(lx), len(rx)))
#         if (mx < 8000 and self.left_fit.any() and self.right_fit.any()):
#             lfit, rfit = self.left_fit, self.right_fit
#             lx, ly, rx, ry = self.lx, self.ly, self.rx, self.ry
#         elif (mn < mx // 2):
#             if len(lx) < len(rx):
#                 lx = (rx - bottom_width)
#                 ly = ry
#                 lfit = np.polyfit(ly, lx, 2)
#             else:
#                 rx = (lx + bottom_width)
#                 ry = ly
#                 rfit = np.polyfit(ry, rx, 2)
#         return lfit, rfit, lx, ly, rx, ry
#
#     def choose_fit(self, im):
#         thresh = threshold_pipeline(im)
#         warped, src, dst = warp_image(thresh)
#         out_img = np.dstack((warped, warped, warped)) * 255
#
#         lfit, rfit, lx, ly, rx, ry, lb, rb = search_for_fit(warped, out_img)
#         # lfit, rfit, lx, ly, rx, ry = search_around_fit(warped, lfit, rfit)
#         self.olx, self.oly, self.orx, self.ory = lx, ly, rx, ry
#
#         bottom_width = rb - lb
#         # lfit, rfit, lx, ly, rx, ry = self.average_fit(lfit, rfit, lx, ly, rx, ry)
#         lfit, rfit, lx, ly, rx, ry = self.validate_fit(lfit, rfit, lx, ly, rx, ry, bottom_width)
#
#         self.left_fit, self.right_fit = lfit, rfit
#         self.lx, self.ly = lx, ly
#         self.rx, self.ry = rx, ry
#
#     def annotate_image(self, im):
#         lfit, rfit, lx, ly, rx, ry = self.left_fit, self.right_fit, self.lx, self.ly, self.rx, self.ry
#         olx, oly, orx, ory = self.olx, self.oly, self.orx, self.ory
#         mask = lane_mask(im, lfit, rfit)
#
#         # Combine the result with the original image
#         result = cv2.addWeighted(im, 1, mask, 0.3, 0)
#
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         h, w, _ = im.shape
#
#         ave_fit = np.average((lfit, rfit), axis=0, weights=(len(lx), len(rx)))
#         yp = np.array(range(h))
#         xp = np.dot(ave_fit, np.array([[y ** 2, y, 1] for y in yp]).T)
#         real_ave_fit = real_fit(xp, yp)
#         rc = radius_of_curvature(real_ave_fit, h)
#
#         real_lfit, real_rfit = real_fit(olx, oly), real_fit(orx, ory)
#         rcl, rcr = radius_of_curvature(real_lfit, h), radius_of_curvature(real_rfit, h)
#         conf = str(len(olx)) + " - " + str(len(orx))
#
#         off = offset_from_centre(w / 2, h, real_lfit, real_rfit)
#         cv2.putText(result, "Radius of curvature : " + str(rc), (100, 50), font, 1.5, (255, 255, 255), 2)
#         cv2.putText(result, "Offset from centre  : " + str(off), (100, 100), font, 1.5, (255, 255, 255), 2)
#         # cv2.putText(result, "Confidence          : " + str(conf), (100,150), font, 1.5, (255,255,255),2)
#         # cv2.putText(result, "Radius of curvature : " + str(rcl), (100,200), font, 1.5, (255,255,255),2)
#         # cv2.putText(result, "Radius of curvature : " + str(rcr), (100,250), font, 1.5, (255,255,255),2)
#
#         return result
#
#     def annotate_pipeline(self, image):
#         self.choose_fit(image)
#         return self.annotate_image(image)
#
#
# im = export_images[8]
# # plt.figure(figsize=(30,20))
# # plt.imshow(im)
# # plt.show()
# plt.figure(figsize=(30, 20))
# ann = Annotator()
# result = ann.annotate_pipeline(im)
# # result = annotate_image(im)
# plt.imshow(result)
# plt.show()
#
# export_images = [mpimg.imread(f) for f in glob.glob(os.path.join("exported", "frame103?.jpeg"))]
# ann = Annotator()
# annotated = [ann.annotate_pipeline(im) for im in export_images]
# display_images(annotated)
# annotated = [annotate_image(im) for im in export_images]
# display_images(annotated)
#
# from moviepy.editor import VideoFileClip
# from IPython.display import HTML
#
# video_output = 'see.mp4'
# clip1 = VideoFileClip("project_video.mp4")
# white_clip = clip1.fl_image(Annotator().annotate_pipeline)
# white_clip.write_videofile(white_output, audio=False)
