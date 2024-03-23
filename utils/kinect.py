from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import ctypes
import math
import time
import copy
import keyboard


class Kinect(object):

    def __init__(self):
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth | PyKinectV2.FrameSourceTypes_Infrared)
        self.color_frame = None
        self.depth_frame = None
        self.infrared_frame = None
        self.w_color = 1920
        self.h_color = 1080
        self.w_depth = 512
        self.h_depth = 424
        self.csp_type = _ColorSpacePoint * np.int(self.w_color * self.h_color)
        self.csp = ctypes.cast(self.csp_type(), ctypes.POINTER(_DepthSpacePoint))
        self.color = None
        self.depth = None
        self.infrared = None
        self.first_time = True

    # Copying this image directly in python is not as efficient as getting another frame directly from C++
    """Get the latest color data"""
    def get_the_last_color(self):
        if self._kinect.has_new_color_frame():
            # the obtained image data is 2d and needs to be converted to the desired format
            frame = self._kinect.get_last_color_frame()
            # return 4 channels, and one channel is not registered
            gbra = frame.reshape([self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4])
            # remove color image data, the default is that the mirror image needs to be flipped
            color_frame = gbra[..., :3][:, ::-1, :]
            return color_frame

    """Get the latest depth data"""
    def get_the_last_depth(self):
        if self._kinect.has_new_depth_frame():
            frame = self._kinect.get_last_depth_frame()
            depth_frame_all = frame.reshape([self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width])
            self.depth_frame = depth_frame_all[:, ::-1]
            return self.depth_frame

    """Get the latest infrared data"""
    def get_the_last_infrared(self, Infrared_threshold = 16000):
        if self._kinect.has_new_infrared_frame():
            frame = self._kinect.get_last_infrared_frame()
            image_infrared_all = frame.reshape([self._kinect.infrared_frame_desc.Height, self._kinect.infrared_frame_desc.Width])
            # image_infrared_all[image_infrared_all > Infrared_threshold] = 0
            # image_infrared_all = image_infrared_all / Infrared_threshold * 255
            self.infrared_frame = image_infrared_all[:, ::-1]
            return self.infrared_frame

    """Match the depth pixels into the color image"""
    def map_depth_points_to_color_points(self, depth_points):
        depth_points = [self.map_depth_point_to_color_point(x) for x in depth_points]
        return depth_points

    def map_depth_point_to_color_point(self, depth_point):
        global valid
        depth_point_to_color = copy.deepcopy(depth_point)
        n = 0
        while 1:
            self.get_the_last_depth()
            self.get_the_last_color()
            color_point = self._kinect._mapper.MapDepthPointToColorSpace(_DepthSpacePoint(511 - depth_point_to_color[1], depth_point_to_color[0]), self.depth_frame[depth_point_to_color[0], 511 - depth_point_to_color[1]])
            if math.isinf(float(color_point.y)):
                n += 1
                if n >= 10000:
                    color_point = [0, 0]
                    break
            else:
                color_point = [np.int0(color_point.y), 1920 - np.int0(color_point.x)]  # image coordinates, human eye Angle
                valid += 1
                print('valid number：', valid)
                break
        return color_point

    """Map an array of color pixels into a depth image"""
    def map_color_points_to_depth_points(self, color_points):
        self.get_the_last_depth()
        self.get_the_last_color()
        self._kinect._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512 * 424), self._kinect._depth_frame_data, ctypes.c_uint(1920 * 1080), self.csp)
        depth_points = [self.map_color_point_to_depth_point(x, True) for x in color_points]
        return depth_points

    def map_color_point_to_depth_point(self, color_point, if_call_flg=False):
        n = 0
        color_point_to_depth = copy.deepcopy(color_point)
        color_point_to_depth[1] = 1920 - color_point_to_depth[1]
        while 1:
            self.get_the_last_depth()
            self.get_the_last_color()
            if not if_call_flg:
                self._kinect._mapper.MapColorFrameToDepthSpace(ctypes.c_uint(512 * 424), self._kinect._depth_frame_data, ctypes.c_uint(1920 * 1080), self.csp)
            if math.isinf(float(self.csp[color_point_to_depth[0] * 1920 + color_point_to_depth[1] - 1].y)) \
                    or np.isnan(self.csp[color_point_to_depth[0] * 1920 + color_point_to_depth[1] - 1].y):
                n += 1
                if n >= 10000:
                    print('Color mapping depth, invalid points')
                    depth_point = [0, 0]
                    break
            else:
                try:
                    depth_point = [np.int0(self.csp[color_point_to_depth[0] * 1920 + color_point_to_depth[1] - 1].y),
                                   np.int0(self.csp[color_point_to_depth[0] * 1920 + color_point_to_depth[1] - 1].x)]
                except OverflowError as e:
                    print('Color mapping depth, invalid points')
                    depth_point = [0, 0]
                break
        depth_point[1] = 512 - depth_point[1]
        return depth_point

    """Get the latest color and depth images as well as infrared images"""
    def get_the_data_of_color_depth_infrared_image(self):
        time_s = time.time()
        if self.first_time:
            while 1:
                n = 0
                self.color = self.get_the_last_color()
                n += 1

                self.depth = self.get_the_last_depth()
                n += 1

                if self._kinect.has_new_infrared_frame():
                    frame = self._kinect.get_last_infrared_frame()
                    image_infrared_all = frame.reshape([self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width])
                    self.infrared = image_infrared_all[:, ::-1]
                    n += 1

                t = time.time() - time_s
                if n == 3:
                    self.first_time = False
                    break
                elif t > 5:
                    print('No image data is obtained, please check that the Kinect2 connection is normal')
                    break
        else:
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                gbra = frame.reshape([self._kinect.color_frame_desc.Height, self._kinect.color_frame_desc.Width, 4])
                gbr = gbra[:, :, :3][:, ::-1, :]
                self.color = gbr

            if self._kinect.has_new_depth_frame():
                frame = self._kinect.get_last_depth_frame()
                image_depth_all = frame.reshape([self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width])
                depth = image_depth_all[:, ::-1]
                self.depth = depth

            if self._kinect.has_new_infrared_frame():
                frame = self._kinect.get_last_infrared_frame()
                image_infrared_all = frame.reshape([self._kinect.depth_frame_desc.Height, self._kinect.depth_frame_desc.Width])
                self.infrared = image_infrared_all[:, ::-1]

        return self.color, self.depth, self.infrared


if __name__ == '__main__':
    i = 1
    kinect = Kinect()
    s = time.time()

    while 1:
        data = kinect.get_the_data_of_color_depth_infrared_image()
        img = data[0]
        mat_intri = np.load('./data/intrinsic_matrix.npy')
        coff_dis = np.load('./data/distortion_cofficients.npy')
        h, w = img.shape[0], img.shape[1]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mat_intri, coff_dis, (w, h), 1, (w, h))
        dst = cv.undistort(img, mat_intri, coff_dis, None, newcameramtx)
        dst = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
        plt.imshow(dst/255)
        plt.show()

        """
        # store the mapping matrix in an npy file
        color_points = np.zeros((512 * 424, 2), dtype=np.int)  # valid number: 207662
        k = 0
        for i in range(424):
            for j in range(512):
                color_points[k] = [i, j]
                k += 1
        depth_map_color = kinect.map_depth_points_to_color_points(color_points)

        # turn to 0 that is not in the mapping range
        depth_map_color[..., 0] = np.where(depth_map_color[..., 0] >= 1080, 0, depth_map_color[..., 0])
        depth_map_color[..., 0] = np.where(depth_map_color[..., 0] < 0, 0, depth_map_color[..., 0])
        depth_map_color[..., 1] = np.where(depth_map_color[..., 1] >= 1920, 0, depth_map_color[..., 1])
        depth_map_color[..., 1] = np.where(depth_map_color[..., 1] < 0, 0, depth_map_color[..., 1])
        
        # interpolated fill 0 values
        zeros = np.array(list(set(np.where(depth_map_color == 0)[0])))
        for zero in zeros:
            if zero < 40 * 512 or zero > 360 * 512:
                continue
            j = 1
            while depth_map_color[zero - j].any() == 0 or depth_map_color[zero + j].any() == 0:
                j += 1
            depth_map_color[zero][0] = (depth_map_color[zero - j][0] + depth_map_color[zero + j][0]) // 2
            depth_map_color[zero][1] = (depth_map_color[zero - j][1] + depth_map_color[zero + j][1]) // 2
        np.save('full_depth_map_color.npy', full_depth_map_color)
        """

        depth_map_color = np.load('./data/full_depth_map_color.npy')   # (424*512, 2)
        full_depth_map_color = depth_map_color
        map_color = dst[full_depth_map_color[..., 0], full_depth_map_color[..., 1]]  # (424*512, 2)
        map_color = map_color.reshape((424, 512, 3))
        plt.imshow(map_color/255)
        plt.show()

        if keyboard.is_pressed('esc'):
            break



