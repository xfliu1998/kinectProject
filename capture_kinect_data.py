import os
import time
import datetime
import glob as gb
import h5py
import keyboard
import cv2
import numpy as np
from utils.calibration import Calibrator
from utils.kinect import Kinect


def get_chess_image(image_num=20):
    out_path = './data/chess/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    camera = cv2.VideoCapture(0)
    i = 0
    while 1:
        (grabbed, img) = camera.read()
        cv2.imshow('img', img)
        if cv2.waitKey(1) & keyboard.is_pressed('space'):  # press space to save an image
            i += 1
            firename = str(f'{out_path}img{str(i)}.jpg')
            cv2.imwrite(firename, img)
            print('write: ', firename)
        if cv2.waitKey(1) & 0xFF == ord('q') or i == image_num:  # press q to finish
            break


def camera_calibrator(shape_inner_corner=(11, 8), size_grid=0.025):
    '''
    :param shape_inner_corner: checkerboard size = 12*9
    :param size_grid: the length of the sides of the checkerboard = 25mm
    '''
    chess_dir = "./data/chess"
    out_path = "./data/dedistortion/chess"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    # create calibrator
    calibrator = Calibrator(chess_dir, shape_inner_corner, size_grid)
    # calibrate the camera
    mat_intri, coff_dis = calibrator.calibrate_camera()
    np.save('./data/intrinsic_matrix.npy', mat_intri)
    np.save('./data/distortion_cofficients.npy', coff_dis)
    print("intrinsic matrix: \n {}".format(mat_intri))
    print("distortion cofficients: \n {}".format(coff_dis))  # (k_1, k_2, p_1, p_2, k_3)
    # dedistortion
    calibrator.dedistortion(chess_dir, out_path)
    return mat_intri, coff_dis


def capture_image(name):
    file_name = './data/h5/' + name + '.h5'
    if os.path.exists(file_name):
        os.remove(file_name)
    open(file_name, "x")
    f = h5py.File(file_name, 'a')
    i = 1
    kinect = Kinect()

    time.sleep(1)
    s = time.time()
    while 1:
        data = kinect.get_the_data_of_color_depth_infrared_image()
        # save data
        f.create_dataset(str(i), data=data[0])
        f.create_dataset(str(i+1), data=data[1])
        i += 2
        cv2.imshow('kinect', data[0])
        cv2.waitKey(1)
        if keyboard.is_pressed('esc'):  # press ESC to exit
            break
    print('record time: %f s' % (time.time() - s))
    return file_name


def dedistortion(mat_intri, coff_dis, img):
    h, w = img.shape[0], img.shape[1]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat_intri, coff_dis, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mat_intri, coff_dis, None, newcameramtx)
    return dst


def save_image(data, name, type, dir='test'):
    global num
    idx = str(num).zfill(6)
    if dir == 'raw':
        color_path = './data/' + name + '/color'
        depth_path = './data/' + name + '/depth'
    else:
        color_path = "./test_data/" + name + '/color'
        depth_path = "./test_data/" + name + '/depth'
    if not os.path.exists(color_path):
        os.makedirs(color_path)
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)
    if type == 'color':
        cv2.imwrite(color_path + '/color-' + idx + '.png', data)
    else:
        cv2.imwrite(depth_path + '/depth-' + idx + '.png', data)
        if dir == 'test':
            num += 1


def center_crop(img,  crop_size):
    tw, th = crop_size
    h, w = img.shape[0], img.shape[1]
    if len(img.shape) == 2:
        crop_img = img[(h - th) // 2:(h + th) // 2, (w - tw) // 2:(w + tw) // 2]
    else:
        crop_img = img[(h - th) // 2:(h + th) // 2, (w - tw) // 2:(w + tw) // 2, :]
    return crop_img


def match_color_depth(color, depth):
    # crop+resize is worse
    full_depth_map_color = np.load('data/full_depth_map_color.npy')
    map_color = color[full_depth_map_color[..., 0], full_depth_map_color[..., 1]]  # (424*512, 2)
    map_color = map_color.reshape((424, 512, 3))
    # 512 * 424
    color = center_crop(map_color, (480, 360))
    depth = center_crop(depth, (480, 360))
    # plt.subplot(1, 2, 1)
    # plt.imshow(color)
    # plt.subplot(1, 2, 2)
    # plt.imshow(depth)
    # plt.show()
    return color, depth


def trans_video(image_path, video_name, fps, res, type):
    img_path = gb.glob(image_path + "/*.png")
    videoWriter = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, res)
    for path in img_path:
        img = cv2.imread(path)
        img = cv2.resize(img, res)
        videoWriter.write(img)
    print('transform ' + type + ' video done!')


def save_video(name):
    currentdate = datetime.datetime.now()
    file_name = str(currentdate.day) + "." + str(currentdate.month) + "." + str(currentdate.hour) + "." + str(currentdate.minute)
    color_path = './data/' + name + '/color'
    # depth_path = './data/' + name + '/depth'
    video_path = './data/' + name + '/video'
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    trans_video(color_path, video_path + '/color-' + file_name + '.avi', 30, (1920, 1080), 'color')
    # trans_video(depth_path, depth_path + '/depth-' + file_name + '.avi', 30, (512, 424), 'depth')


if __name__ == '__main__':
    # 1. shooting calibration images
    get_chess_image()

    # 2. camera calibration
    mat_intri, coff_dis = camera_calibrator()
    # mat_intri = np.load('./data/intrinsic_matrix.npy')
    # coff_dis = np.load('./data/distortion_cofficients.npy')

    # 3. capture object images to save h5 file
    name = 'object'
    file_name = capture_image(name)

    f = h5py.File(file_name, 'r')
    num = 0
    for i in range(1, len(f.keys()), 2):
        color = f[str(i)][:]
        depth = f[str(i + 1)][:]

        # 4. data process: dedistortion; match color and depth images; save color/depth images
        dedistortion_color = dedistortion(mat_intri, coff_dis, color)
        save_image(dedistortion_color, name, 'color', 'raw')
        save_image(depth, name, 'depth', 'raw')
        new_color, new_depth = match_color_depth(dedistortion_color, depth)
        save_image(new_color, name, 'color', 'test')
        save_image(new_depth, name, 'depth', 'test')
    f.close()
    print('image save done!')

    # 5. convert to video
    save_video(name)









