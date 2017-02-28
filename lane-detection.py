import glob
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import numpy as np

dir = "camera_cal"
images = [mpimg.imread(f) for f in glob.glob(os.path.join(dir, "calibration*.jpg"))]


def display_images(images, cmap=None, col=4, title=None):
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

def calibrateCamera(images):
    r, c = (9,6)
    processed = np.vstack([findChessboardCorners(im, (r, c)) for im in images])
    valid = processed[processed[:,1]==True]
    image_shape = valid[0, 0].shape
    image_points = np.array([v.squeeze() for v in valid[:,2]])
    coords = np.zeros((r*c,3), np.float32)
    coords[:,:2] = np.mgrid[:r, :c].T.reshape(-1, 2)
    object_points = np.tile(coords, (image_points.shape[0], 1, 1))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_shape[0:2], None, None)
    return ret, mtx, dist, rvecs, tvecs, valid



# Calibrate
ret, mtx, dist, rvecs, tvecs, valid = calibrateCamera(images)
#display_images(valid[:,0])

# Undistort
sample_image = images[0]
dst = cv2.undistort(sample_image, mtx, dist, None, mtx)
plt.imshow(sample_image)
plt.figure()
plt.imshow(dst)
plt.show()


def extract_channels(im):
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    yuv = cv2.cvtColor(im, cv2.COLOR_RGB2YUV)
    gg, r, g, b, h, l, s, hh, ss, vv, y, u, yv = [
        gray, im[:,:,0], im[:,:,1], im[:,:,2],
        hls[:,:,0], hls[:,:,1], hls[:,:,2],
        hsv[:,:,0], hsv[:,:,1], hsv[:,:,2],
        yuv[:,:,0], yuv[:,:,1], yuv[:,:,2]]
    return gg, r, g, b, h, l, s, hh, ss, vv, y, u, yv

test_dir = "test_images"
straight_lines1 = mpimg.imread(os.path.join(test_dir, "straight_lines1.jpg"))
straight_lines2 = mpimg.imread(os.path.join(test_dir, "straight_lines2.jpg"))
channels = extract_channels(straight_lines1)
display_images(channels[0:1], 'gray', col=1, title="Gray")
display_images(channels[1:4], 'gray', col=3, title="RGB")
display_images(channels[4:7], 'gray', col=3, title="HLS")
display_images(channels[7:10], 'gray', col=3, title="HSV")
display_images(channels[10:13], 'gray', col=3, title="YUV")

test_images = [mpimg.imread(f) for f in glob.glob(os.path.join(test_dir, "test*.jpg"))]
display_images(test_images, col=3, title="Test images")
l_channels = [extract_channels(im)[5] for im in test_images]
s_channels = [extract_channels(im)[6] for im in test_images]
display_images(l_channels, col=3, title="L channel only", cmap='gray')
display_images(s_channels, col=3, title="S channel only", cmap='gray')


def simple_norm(im):
    return im / np.max(im)


def threshold(im, mn, mx):
    im = im.astype(np.float32)
    norm = simple_norm(im)
    fltr = np.logical_and(norm >= mn, norm <= mx)
    # fltr = np.logical_and(im >= mn*255, im <= mx*255)
    thresh = np.zeros_like(im, dtype=np.float32)
    # thresh[fltr] = 1
    thresh[fltr] = norm[fltr]
    return thresh


def threshold_color(im):
    channels = extract_channels(im)
    _, r, g, b, h, l, s, hh, ss, vv, y, u, yv = channels
    thresholded_r = threshold(r, 0.75, 1.0)
    thresholded_s = threshold(s, 0.75, 1.0)
    thresholded_l = threshold(l, 0.90, 1.0)
    thresholded_ss = threshold(ss, 0.70, 1.0)
    thresholded_vv = threshold(vv, 0.90, 1.0)
    thresholded_y = threshold(y, 0.90, 1.0)
    return thresholded_s, thresholded_l, thresholded_r, thresholded_vv, thresholded_y


def gradients(img, ksize=13):
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,1]
    gray = np.uint8(img)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    direction = np.arctan2(abs_sobel_y, abs_sobel_x)
    return abs_sobel_x, abs_sobel_y, magnitude, direction


def threshold_gradients(im):
    x_grad, y_grad, mag_grad, dir_grad = gradients(im)
    x_grad = threshold(x_grad, 0.3, 1.0)
    y_grad = threshold(y_grad, 0.3, 1.0)
    mag_grad = threshold(mag_grad, 0.3, 1.0)
    dir_grad = threshold(dir_grad, 0.5, 0.8)
    return x_grad, y_grad, mag_grad, dir_grad


def threshold_pipeline(im):
    undist = cv2.undistort(im, mtx, dist, None, mtx)
    thresh_s, thresh_l, thresh_r, thresh_vv, thresh_y = threshold_color(im)
    color_combined = np.logical_or(thresh_s, thresh_l)
    x_grad, y_grad, mag_grad, dir_grad = threshold_gradients(color_combined)
    grad_combined = np.logical_and(x_grad, y_grad)
    dir_combined = np.logical_and(dir_grad, mag_grad)
    combined = np.logical_or(grad_combined, dir_combined)

    # thresh_image = threshold(combined, 0.3, 1.0)
    thresh_image = np.dstack((x_grad, y_grad, dir_grad))
    return combined


# thresh_images, thresh_s, thresh_color, combined_grad = zip(*[threshold_pipeline(im) for im in test_images])
# display_images(thresh_color, col=3, cmap='gray', title='Combined Channels')
# # diff = [c - s for c, s in zip(thresh_color, thresh_s)]
# # display_images(diff, col=3, cmap='gray', title='Diff Channels')
# display_images(combined_grad, col=3, cmap='gray', title="Combined gradient")
# display_images(thresh_images, col=3, cmap='gray', title="Thresholded")

thresh_images = [threshold_pipeline(im) for im in test_images]
# display_images(test_images, col=3)
display_images(thresh_images, cmap='gray', col=3)


def mask_image(im):
    im = np.uint8(im)
    mask = np.zeros_like(im)
    vertices = np.array([[[180, 720], [570, 450], [710, 450], [1190, 720]]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(mask, im)


def warp_image(im):
    # src = np.float32([[190,720],[580,450],[700,450],[1170,720]])
    src = np.float32([[200, 720], [590, 450], [690, 450], [1100, 720]])
    dst = np.float32([[200, 720], [200, 0], [1080, 0], [1080, 720]])
    maxy, maxx = im.shape
    M = cv2.getPerspectiveTransform(src, dst)
    unsigned = np.uint8(im)
    warped = cv2.warpPerspective(unsigned, M, (maxx, maxy), flags=cv2.INTER_LINEAR)
    return warped, src, dst


def unwarp_image(im, src, dst):
    maxy, maxx = im.shape[0], im.shape[1]
    unsigned = np.uint8(im)
    Minv = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(unsigned, Minv, (maxx, maxy), flags=cv2.INTER_LINEAR)
    return unwarped


def plot_warped(im, warped, src, dst):
    plt.imshow(im, cmap='gray')
    plt.title("Transform")
    plt.plot(*zip(src[0], src[1]), 'r-')
    plt.plot(*zip(src[2], src[3]), 'r-')
    plt.figure()
    plt.imshow(warped, cmap='gray')
    plt.title("Warped")
    plt.plot(*zip(dst[0], dst[1]), 'r-')
    plt.plot(*zip(dst[2], dst[3]), 'r-')
    plt.show()


def plot_fit(out_img, left_fit, right_fit, color='yellow', title='Line fit'):
    h, w, _ = out_img.shape
    y_values = np.arange(h)
    coeff = np.array([[y ** 2, y, 1] for y in y_values])
    lx_values = np.dot(left_fit, coeff.T)
    rx_values = np.dot(right_fit, coeff.T)

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

    plt.imshow(result)
    plt.title(title)
    plt.plot(lx_values, y_values, color=color)
    plt.plot(rx_values, y_values, color=color)
    plt.xlim(0, 1280)
    plt.ylim(720, 0)

    plt.show()


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


def search_for_fit(warped, out_img):
    h, w = warped.shape
    top, bottom = warped[:h // 2, :], warped[h // 2:, :]
    hist_bottom = np.sum(bottom, axis=0)
    left_bottom, right_bottom = np.argmax(hist_bottom[:w // 2]), w // 2 + np.argmax(hist_bottom[w // 2:])

    lxp, lyp = slide_windows(warped, out_img, left_bottom, 9, 100)
    rxp, ryp = slide_windows(warped, out_img, right_bottom, 9, 100)
    out_img[lyp, lxp] = [255, 0, 0]
    out_img[ryp, rxp] = [0, 0, 255]

    left_fit = np.polyfit(lyp, lxp, 2)
    right_fit = np.polyfit(ryp, rxp, 2)

    return left_fit, right_fit, lxp, lyp, rxp, ryp, left_bottom, right_bottom


def search_around_fit(im, lfit, rfit, margin=50):
    nzy, nzx = np.nonzero(im)
    coeff = np.array([[y ** 2, y, 1] for y in nzy])
    plx = np.dot(lfit, coeff.T)
    prx = np.dot(rfit, coeff.T)
    lcond = (nzx - margin < plx) & (nzx + margin > plx)
    rcond = (nzx - margin < prx) & (nzx + margin > prx)
    nlfit = np.polyfit(nzy[lcond], nzx[lcond], 2)
    nrfit = np.polyfit(nzy[rcond], nzx[rcond], 2)
    lx, ly, rx, ry = nzx[lcond], nzy[lcond], nzx[rcond], nzy[rcond]
    return nlfit, nrfit, lx, ly, rx, ry


def real_fit(x, y):
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    fit = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
    return fit


def radius_of_curvature(fit, y):
    a, b, c = fit
    return ((1 + (2 * a * y + b) ** 2) ** (3 / 2)) / np.absolute(2 * a)


# im = thresh_images[2]
im = threshold_pipeline(export_images[8])

warped, src, dst = warp_image(im)
plot_warped(im, warped, src, dst)

out_img = np.dstack((warped, warped, warped)) * 255
lfit, rfit, lx, ly, rx, ry, _, _ = search_for_fit(warped, out_img)
real_lfit = real_fit(lx, ly)
real_rfit = real_fit(rx, ry)
plt.imshow(out_img)
plt.title("Sliding windows")
plt.show()

out_img = np.dstack((warped, warped, warped)) * 255
plot_fit(out_img, lfit, rfit, title='Sliding window fit')
nlfit, nrfit, nlx, nly, nrx, nry = search_around_fit(warped, lfit, rfit)
plot_fit(out_img, nlfit, nrfit, title='Smoother fit')

rl, rr = radius_of_curvature(real_lfit, im.shape[0]), radius_of_curvature(real_rfit, im.shape[0])
print(rl, 'm ', rr, 'm ')

print(lfit, rfit)
print(nlfit, nrfit)
print(len(lx), len(ly), len(rx), len(ry))
print(len(nlx), len(nly), len(nrx), len(nry))


def offset_from_centre(x, y, real_lfit, real_rfit):
    ym_per_pix = 30 / 720
    xm_per_pix = 3.7 / 700
    xm, ym = x * xm_per_pix, y * ym_per_pix
    coeff = [ym ** 2, ym, 1]
    lx = np.dot(real_lfit, coeff)
    rx = np.dot(real_rfit, coeff)
    return xm - (rx + lx) / 2


def lane_mask(im, lfit, rfit):
    h, w, _ = im.shape
    y_values = np.arange(h)
    coeff = np.array([[y ** 2, y, 1] for y in y_values])
    lx_values = np.dot(lfit, coeff.T)
    rx_values = np.dot(rfit, coeff.T)

    color_warp = np.zeros_like(im).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([lx_values, y_values]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([rx_values, y_values])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    return unwarp_image(color_warp, src, dst)


def annotate_image(im):
    thresh = threshold_pipeline(im)
    warped, src, dst = warp_image(thresh)
    out_img = np.dstack((warped, warped, warped)) * 255
    lfit, rfit, lx, ly, rx, ry, _, _ = search_for_fit(warped, out_img)
    # lfit, rfit, lx, ly, rx, ry = search_around_fit(warped, lfit, rfit)

    mask = lane_mask(im, lfit, rfit)

    # Combine the result with the original image
    result = cv2.addWeighted(im, 1, mask, 0.3, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w, _ = im.shape

    real_lfit, real_rfit = real_fit(lx, ly), real_fit(rx, ry)
    rcl, rcr = radius_of_curvature(real_lfit, h), radius_of_curvature(real_rfit, h)
    off = offset_from_centre(w / 2, h, real_lfit, real_rfit)
    conf = str(len(lx)) + " - " + str(len(rx))
    cv2.putText(result, "Radius (Left)      : " + str(rcl), (100, 50), font, 1.5, (255, 255, 255), 2)
    cv2.putText(result, "Radius (Right)     : " + str(rcr), (100, 100), font, 1.5, (255, 255, 255), 2)
    cv2.putText(result, "Offset from centre : " + str(off), (100, 150), font, 1.5, (255, 255, 255), 2)
    cv2.putText(result, "Confidence         : " + str(conf), (100, 200), font, 1.5, (255, 255, 255), 2)

    return result


class Annotator:
    def __init__(self):
        self.left_fit = np.array([])
        self.right_fit = np.array([])
        self.lx = []
        self.ly = []
        self.rx = []
        self.ry = []
        self.olx = []
        self.oly = []
        self.orx = []
        self.ory = []

    def average_fit(self, lfit, rfit, lx, ly, rx, ry):
        a, b, _ = np.average((lfit, rfit), axis=0, weights=(len(lx), len(rx)))
        nlfit = np.array([a, b, lfit[-1]])
        nrfit = np.array([a, b, rfit[-1]])
        lcoeff = np.array([[y ** 2, y, 1] for y in ly])
        rcoeff = np.array([[y ** 2, y, 1] for y in ry])
        nlx = np.dot(nlfit, lcoeff.T)
        nrx = np.dot(nrfit, rcoeff.T)
        return nlfit, nrfit, nlx, ly, nrx, ry

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

    def choose_fit(self, im):
        thresh = threshold_pipeline(im)
        warped, src, dst = warp_image(thresh)
        out_img = np.dstack((warped, warped, warped)) * 255

        lfit, rfit, lx, ly, rx, ry, lb, rb = search_for_fit(warped, out_img)
        # lfit, rfit, lx, ly, rx, ry = search_around_fit(warped, lfit, rfit)
        self.olx, self.oly, self.orx, self.ory = lx, ly, rx, ry

        bottom_width = rb - lb
        # lfit, rfit, lx, ly, rx, ry = self.average_fit(lfit, rfit, lx, ly, rx, ry)
        lfit, rfit, lx, ly, rx, ry = self.validate_fit(lfit, rfit, lx, ly, rx, ry, bottom_width)

        self.left_fit, self.right_fit = lfit, rfit
        self.lx, self.ly = lx, ly
        self.rx, self.ry = rx, ry

    def annotate_image(self, im):
        lfit, rfit, lx, ly, rx, ry = self.left_fit, self.right_fit, self.lx, self.ly, self.rx, self.ry
        olx, oly, orx, ory = self.olx, self.oly, self.orx, self.ory
        mask = lane_mask(im, lfit, rfit)

        # Combine the result with the original image
        result = cv2.addWeighted(im, 1, mask, 0.3, 0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        h, w, _ = im.shape

        ave_fit = np.average((lfit, rfit), axis=0, weights=(len(lx), len(rx)))
        yp = np.array(range(h))
        xp = np.dot(ave_fit, np.array([[y ** 2, y, 1] for y in yp]).T)
        real_ave_fit = real_fit(xp, yp)
        rc = radius_of_curvature(real_ave_fit, h)

        real_lfit, real_rfit = real_fit(olx, oly), real_fit(orx, ory)
        rcl, rcr = radius_of_curvature(real_lfit, h), radius_of_curvature(real_rfit, h)
        conf = str(len(olx)) + " - " + str(len(orx))

        off = offset_from_centre(w / 2, h, real_lfit, real_rfit)
        cv2.putText(result, "Radius of curvature : " + str(rc), (100, 50), font, 1.5, (255, 255, 255), 2)
        cv2.putText(result, "Offset from centre  : " + str(off), (100, 100), font, 1.5, (255, 255, 255), 2)
        # cv2.putText(result, "Confidence          : " + str(conf), (100,150), font, 1.5, (255,255,255),2)
        # cv2.putText(result, "Radius of curvature : " + str(rcl), (100,200), font, 1.5, (255,255,255),2)
        # cv2.putText(result, "Radius of curvature : " + str(rcr), (100,250), font, 1.5, (255,255,255),2)

        return result

    def annotate_pipeline(self, image):
        self.choose_fit(image)
        return self.annotate_image(image)


im = export_images[8]
# plt.figure(figsize=(30,20))
# plt.imshow(im)
# plt.show()
plt.figure(figsize=(30, 20))
ann = Annotator()
result = ann.annotate_pipeline(im)
# result = annotate_image(im)
plt.imshow(result)
plt.show()

export_images = [mpimg.imread(f) for f in glob.glob(os.path.join("exported", "frame103?.jpeg"))]
ann = Annotator()
annotated = [ann.annotate_pipeline(im) for im in export_images]
display_images(annotated)
annotated = [annotate_image(im) for im in export_images]
display_images(annotated)

from moviepy.editor import VideoFileClip
from IPython.display import HTML

video_output = 'see.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(Annotator().annotate_pipeline)
white_clip.write_videofile(white_output, audio=False)
