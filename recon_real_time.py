import os
import time
import math
import numpy as np
from skimage import io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import cv2
import open3d as o3d


def cal_neighbors():
    # preprocessing neighborhood point coordinates
    neighbors_list = dict()
    for r in range(1, 9):
        neighbors = []
        for x in range(-r, r + 1):
            for y in range(-r, r + 1):
                if math.sqrt(x ** 2 + y ** 2) <= r:
                    neighbors.append([x, y])
        neighbors.sort(key=lambda x: x[0] ** 2 + x[1] ** 2)
        neighbors_list[r] = neighbors[1:]
    print(neighbors_list)


neighbors_list = {1: [[-1, 0], [0, -1], [0, 1], [1, 0]],
                  2: [[-1, 0], [0, -1], [0, 1], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1], [-2, 0], [0, -2], [0, 2], [2, 0]],
                  3: [[-1, 0], [0, -1], [0, 1], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1], [-2, 0], [0, -2], [0, 2], [2, 0], [-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1], [-2, -2], [-2, 2], [2, -2], [2, 2], [-3, 0], [0, -3], [0, 3], [3, 0]],
                  4: [[-1, 0], [0, -1], [0, 1], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1], [-2, 0], [0, -2], [0, 2], [2, 0], [-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1], [-2, -2], [-2, 2], [2, -2], [2, 2], [-3, 0], [0, -3], [0, 3], [3, 0], [-3, -1], [-3, 1], [-1, -3], [-1, 3], [1, -3], [1, 3], [3, -1], [3, 1], [-3, -2], [-3, 2], [-2, -3], [-2, 3], [2, -3], [2, 3], [3, -2], [3, 2], [-4, 0], [0, -4], [0, 4], [4, 0]],
                  5: [[-1, 0], [0, -1], [0, 1], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1], [-2, 0], [0, -2], [0, 2], [2, 0], [-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1], [-2, -2], [-2, 2], [2, -2], [2, 2], [-3, 0], [0, -3], [0, 3], [3, 0], [-3, -1], [-3, 1], [-1, -3], [-1, 3], [1, -3], [1, 3], [3, -1], [3, 1], [-3, -2], [-3, 2], [-2, -3], [-2, 3], [2, -3], [2, 3], [3, -2], [3, 2], [-4, 0], [0, -4], [0, 4], [4, 0], [-4, -1], [-4, 1], [-1, -4], [-1, 4], [1, -4], [1, 4], [4, -1], [4, 1], [-3, -3], [-3, 3], [3, -3], [3, 3], [-4, -2], [-4, 2], [-2, -4], [-2, 4], [2, -4], [2, 4], [4, -2], [4, 2], [-5, 0], [-4, -3], [-4, 3], [-3, -4], [-3, 4], [0, -5], [0, 5], [3, -4], [3, 4], [4, -3], [4, 3], [5, 0]],
                  6: [[-1, 0], [0, -1], [0, 1], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1], [-2, 0], [0, -2], [0, 2], [2, 0], [-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1], [-2, -2], [-2, 2], [2, -2], [2, 2], [-3, 0], [0, -3], [0, 3], [3, 0], [-3, -1], [-3, 1], [-1, -3], [-1, 3], [1, -3], [1, 3], [3, -1], [3, 1], [-3, -2], [-3, 2], [-2, -3], [-2, 3], [2, -3], [2, 3], [3, -2], [3, 2], [-4, 0], [0, -4], [0, 4], [4, 0], [-4, -1], [-4, 1], [-1, -4], [-1, 4], [1, -4], [1, 4], [4, -1], [4, 1], [-3, -3], [-3, 3], [3, -3], [3, 3], [-4, -2], [-4, 2], [-2, -4], [-2, 4], [2, -4], [2, 4], [4, -2], [4, 2], [-5, 0], [-4, -3], [-4, 3], [-3, -4], [-3, 4], [0, -5], [0, 5], [3, -4], [3, 4], [4, -3], [4, 3], [5, 0], [-5, -1], [-5, 1], [-1, -5], [-1, 5], [1, -5], [1, 5], [5, -1], [5, 1], [-5, -2], [-5, 2], [-2, -5], [-2, 5], [2, -5], [2, 5], [5, -2], [5, 2], [-4, -4], [-4, 4], [4, -4], [4, 4], [-5, -3], [-5, 3], [-3, -5], [-3, 5], [3, -5], [3, 5], [5, -3], [5, 3], [-6, 0], [0, -6], [0, 6], [6, 0]],
                  7: [[-1, 0], [0, -1], [0, 1], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1], [-2, 0], [0, -2], [0, 2], [2, 0], [-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1], [-2, -2], [-2, 2], [2, -2], [2, 2], [-3, 0], [0, -3], [0, 3], [3, 0], [-3, -1], [-3, 1], [-1, -3], [-1, 3], [1, -3], [1, 3], [3, -1], [3, 1], [-3, -2], [-3, 2], [-2, -3], [-2, 3], [2, -3], [2, 3], [3, -2], [3, 2], [-4, 0], [0, -4], [0, 4], [4, 0], [-4, -1], [-4, 1], [-1, -4], [-1, 4], [1, -4], [1, 4], [4, -1], [4, 1], [-3, -3], [-3, 3], [3, -3], [3, 3], [-4, -2], [-4, 2], [-2, -4], [-2, 4], [2, -4], [2, 4], [4, -2], [4, 2], [-5, 0], [-4, -3], [-4, 3], [-3, -4], [-3, 4], [0, -5], [0, 5], [3, -4], [3, 4], [4, -3], [4, 3], [5, 0], [-5, -1], [-5, 1], [-1, -5], [-1, 5], [1, -5], [1, 5], [5, -1], [5, 1], [-5, -2], [-5, 2], [-2, -5], [-2, 5], [2, -5], [2, 5], [5, -2], [5, 2], [-4, -4], [-4, 4], [4, -4], [4, 4], [-5, -3], [-5, 3], [-3, -5], [-3, 5], [3, -5], [3, 5], [5, -3], [5, 3], [-6, 0], [0, -6], [0, 6], [6, 0], [-6, -1], [-6, 1], [-1, -6], [-1, 6], [1, -6], [1, 6], [6, -1], [6, 1], [-6, -2], [-6, 2], [-2, -6], [-2, 6], [2, -6], [2, 6], [6, -2], [6, 2], [-5, -4], [-5, 4], [-4, -5], [-4, 5], [4, -5], [4, 5], [5, -4], [5, 4], [-6, -3], [-6, 3], [-3, -6], [-3, 6], [3, -6], [3, 6], [6, -3], [6, 3], [-7, 0], [0, -7], [0, 7], [7, 0]],
                  8: [[-1, 0], [0, -1], [0, 1], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1], [-2, 0], [0, -2], [0, 2], [2, 0], [-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1], [-2, -2], [-2, 2], [2, -2], [2, 2], [-3, 0], [0, -3], [0, 3], [3, 0], [-3, -1], [-3, 1], [-1, -3], [-1, 3], [1, -3], [1, 3], [3, -1], [3, 1], [-3, -2], [-3, 2], [-2, -3], [-2, 3], [2, -3], [2, 3], [3, -2], [3, 2], [-4, 0], [0, -4], [0, 4], [4, 0], [-4, -1], [-4, 1], [-1, -4], [-1, 4], [1, -4], [1, 4], [4, -1], [4, 1], [-3, -3], [-3, 3], [3, -3], [3, 3], [-4, -2], [-4, 2], [-2, -4], [-2, 4], [2, -4], [2, 4], [4, -2], [4, 2], [-5, 0], [-4, -3], [-4, 3], [-3, -4], [-3, 4], [0, -5], [0, 5], [3, -4], [3, 4], [4, -3], [4, 3], [5, 0], [-5, -1], [-5, 1], [-1, -5], [-1, 5], [1, -5], [1, 5], [5, -1], [5, 1], [-5, -2], [-5, 2], [-2, -5], [-2, 5], [2, -5], [2, 5], [5, -2], [5, 2], [-4, -4], [-4, 4], [4, -4], [4, 4], [-5, -3], [-5, 3], [-3, -5], [-3, 5], [3, -5], [3, 5], [5, -3], [5, 3], [-6, 0], [0, -6], [0, 6], [6, 0], [-6, -1], [-6, 1], [-1, -6], [-1, 6], [1, -6], [1, 6], [6, -1], [6, 1], [-6, -2], [-6, 2], [-2, -6], [-2, 6], [2, -6], [2, 6], [6, -2], [6, 2], [-5, -4], [-5, 4], [-4, -5], [-4, 5], [4, -5], [4, 5], [5, -4], [5, 4], [-6, -3], [-6, 3], [-3, -6], [-3, 6], [3, -6], [3, 6], [6, -3], [6, 3], [-7, 0], [0, -7], [0, 7], [7, 0], [-7, -1], [-7, 1], [-5, -5], [-5, 5], [-1, -7], [-1, 7], [1, -7], [1, 7], [5, -5], [5, 5], [7, -1], [7, 1], [-6, -4], [-6, 4], [-4, -6], [-4, 6], [4, -6], [4, 6], [6, -4], [6, 4], [-7, -2], [-7, 2], [-2, -7], [-2, 7], [2, -7], [2, 7], [7, -2], [7, 2], [-7, -3], [-7, 3], [-3, -7], [-3, 7], [3, -7], [3, 7], [7, -3], [7, 3], [-6, -5], [-6, 5], [-5, -6], [-5, 6], [5, -6], [5, 6], [6, -5], [6, 5], [-8, 0], [0, -8], [0, 8], [8, 0]]}


def get_mask_from_dist(image, d=1):
    mask = np.where(image[..., -1] <= d, image[..., -1], 0)
    # mask1 = load_mask(data_path + '/mask/0.png')[..., 0]  # (h, w)
    # mask2 = load_mask(data_path + '/mask/1.png')[..., 0]  # (h, w)
    # mask += mask1
    # mask += mask2
    mask = np.tile(np.expand_dims(mask, axis=-1), 3)
    mask_data = np.where(mask, 255, 0)
    return mask_data


def crop(img,  crop_size, crop_type):
    th, tw = crop_size
    h, w = img.shape[0], img.shape[1]
    if crop_type == 'center':
        if len(img.shape) == 2:
            crop_img = img[(h - th) // 2:(h + th) // 2, (w - tw) // 2:(w + tw) // 2]
        else:
            crop_img = img[(h - th) // 2:(h + th) // 2, (w - tw) // 2:(w + tw) // 2, :]
    # down sample: INTER_NEAREST INTER_AREA
    elif crop_type == 'cv2resize':
        crop_img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_NEAREST)
    else:  # INTER_LINEAR
        crop_img = cv2.resize(img, (tw, th))
    return crop_img


def normalize(img, normal_type):
    h, w = img.shape[0], img.shape[1]
    source_color_img = img[:, :, :3]
    source_depth_img = img[:, :, 3:]
    if normal_type == 'standard_scaler':
        color_img = source_color_img.reshape(h * w, 3)
        depth_img = source_depth_img.reshape(h * w, 3)
        std_sca1 = StandardScaler()
        std_sca2 = StandardScaler()
        color_img_sca = std_sca1.fit_transform(color_img)
        depth_img_sca = std_sca2.fit_transform(depth_img)
        color_img_sca = color_img_sca.reshape(h, w, 3)
        depth_img_sca = depth_img_sca.reshape(h, w, 3)
    elif normal_type == 'minMax_scalar':
        color_img = source_color_img.reshape(h * w, 3)
        depth_img = source_depth_img.reshape(h * w, 3)
        mm_sca1 = MinMaxScaler()
        mm_sca2 = MinMaxScaler()
        color_img_sca = mm_sca1.fit_transform(color_img)
        depth_img_sca = mm_sca2.fit_transform(depth_img)
        color_img_sca = color_img_sca.reshape(h, w, 3)
        depth_img_sca = depth_img_sca.reshape(h, w, 3)
    else:
        color_img_sca = source_color_img / 255
        depth_img_sca = source_depth_img
    img_sca = np.concatenate((color_img_sca, depth_img_sca, source_color_img, source_depth_img), axis=-1)
    return img_sca


def backproject_depth(depth_image, intrinsics):
    assert len(depth_image.shape) == 2
    intrinsics[-1][-1] = 1
    height, width = depth_image.shape
    if depth_image.dtype != np.float32:
        depth_image = depth_image.astype(np.float32)

    depth_image /= 1000  # unit: m
    img = np.ones((width, height, 3))
    img[..., 0] = np.array([[i] * height for i in range(width)]).reshape(width, height)
    img[..., 1] = np.array(list(range(height)) * width).reshape(width, height)
    Z = np.repeat(np.transpose(depth_image).reshape(width, height, 1), 3, axis=2)
    img2d = img * Z  # (h, w, 3)
    point_image = np.matmul(img2d, np.linalg.inv(np.transpose(intrinsics)))  # 3, hxw  xyz
    point_image = point_image.swapaxes(0, 1).astype(np.float32)

    """
    point_image = np.zeros((height, width, 3), dtype=np.float32)
    k_x, k_y, u_0, v_0 = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    for v in range(height):  # row -> y
        for u in range(width):  # col -> x
            if depth_image[v, u] == 0:
                continue
            depth = depth_image[v, u]
            z_c = depth / 1000   # unit: m
            x_c = (u - u_0) * z_c / k_x
            y_c = (v - v_0) * z_c / k_y
            point_image[v, u] = np.array([x_c, y_c, z_c], dtype=np.float32)
    """
    return point_image


def load_image(color_image_path, depth_image_path, intrinsics):
    color_image = io.imread(color_image_path)  # (h, w, 3)  RGB
    depth_image = io.imread(depth_image_path)  # (h, w)
    # depth_image = cv2.GaussianBlur(depth_image, (3, 3), 1)
    depth_image = backproject_depth(depth_image, intrinsics)  # (h, w, 3)  xyz
    image = np.concatenate((color_image, depth_image), axis=-1)  # (h, w, 6)
    return image  # (h, w, 6)


def load_mask(mask_image_path):
    mask_image = cv2.imread(mask_image_path)
    return mask_image / 255.  # (h, w, 3)


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])   # red
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])  # gray
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def remove_noisy(image, nb_points=30, radius=0.02):
    height, width = image.shape[:2]
    point = image[..., 3:6]
    point = point.reshape(height * width, 3)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point)
    # convert outlier coordinates to 0
    down_sample_points = 1
    uni_down_src_pc = pc.uniform_down_sample(every_k_points=down_sample_points)
    # delete points that have few neighborhood points around a sphere of a given radius
    cl1, ind = uni_down_src_pc.remove_radius_outlier(nb_points=nb_points, radius=radius)
    # display outlier and inlier
    # display_inlier_outlier(pc, ind)
    ind = set(ind)
    for i in range(height * width):
        if i not in ind:
            point[i] = np.zeros((1, 3), dtype=np.float32)
    point = point.reshape(height, width, 3)
    image[..., 3:6] = point
    image[..., 9:] = point
    return image


def data_process(data):
    intrinsics_path = data['intrinsics_path']
    if intrinsics_path.endswith('txt'):
        intrinsics = []
        with open(intrinsics_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                intrinsics.append(list(map(float, line.split()))[:-1])
        intrinsics = np.array(intrinsics[:-1], dtype=np.float32)
    else:
        intrinsics = np.load(intrinsics_path)

    color_image_path = os.path.join(data_path, data["color"])
    depth_image_path = os.path.join(data_path, data["depth"])
    image = load_image(color_image_path, depth_image_path, intrinsics)

    # mask = get_mask_from_dist(image)  # move_dragon

    # crop and normalize
    crop_size = input_height, input_width
    image = crop(image, crop_size, crop_type)
    image = normalize(image, normal_type='')

    # mask image
    mask = load_mask(data["mask"])  # (h, w)
    mask = crop(mask, crop_size, crop_type)
    mask = np.where(mask, True, False)
    image = np.where(np.tile(mask, 4), image, 0.)   # (h, w, 12)  normal_RGB+normal_xyz+rgb+xyz

    # point cloud denoising
    image = remove_noisy(image)
    return image, mask, intrinsics


def batch_data_process(img_num, data_path):
    pre_data = []
    for i, img_number in enumerate(range(img_num)):
        img_number = f"{img_number:06d}"
        data = {'color': data_path + 'rgbd/frame-' + img_number + '.color.png',
                'depth': data_path + 'rgbd/frame-' + img_number + '.depth.png',
                'mask': data_path + '/mask/mask' + img_number + '.png',
                'intrinsics_path': data_path + 'colorIntrinsics.txt'}
        image, mask, intrinsics = data_process(data)
        # cv2.imwrite(f'{data_path}/mask/mask{img_number}.png', mask)
        pre_data.append([image, mask, intrinsics])
        print(i)
    return pre_data


def map_3d_to_2d(point_set, intrinsics, height, width):
    # x = KX / z_c
    point_rearr = point_set.transpose(1, 0)  # (3, num_points)
    point_proj2D = np.divide(np.matmul(intrinsics, point_rearr), point_rearr[2:, :])
    point_pred = point_proj2D[:2, :].transpose(1, 0)  # (num_points, 2)  u v
    pred_image = np.zeros((height, width, 3), dtype=np.float32)
    alpha = 1.5 if data_type == 'kinect' else 2  # 640/width  480/height
    # normalization, truncation, assignment
    point_pred_int = np.round(point_pred / alpha).astype(np.int)  # (num_points, 2)  u v
    point_pred_int[..., 0] = np.clip(point_pred_int[..., 0], 0, width - 1)
    point_pred_int[..., 1] = np.clip(point_pred_int[..., 1], 0, height - 1)
    pred_image[point_pred_int[..., 1], point_pred_int[..., 0]] = point_set
    return pred_image


def uneven_upsample_based_mask(pred_image, mask_image, radius):
    # pred_image: h*w*3 xyz  mask_image: h*w*3
    up_image = pred_image.copy()
    # the index that is inside mask and the current point is null
    mask_image_inf = np.where(mask_image, 0, float('inf')) + pred_image
    # n1: number of points to be interpolated  2:uv
    idx = np.argwhere(mask_image_inf[..., 0] == 0)  # (n1, 2)
    # n2: neighbor count  2:uv
    neighbors = np.array(neighbors_list[radius])  # (n2, 2)
    n1, n2 = idx.shape[0], neighbors.shape[0]
    idx_expand = np.tile(np.expand_dims(idx, axis=-1), n2).swapaxes(1, 2)
    neighbors_expand = np.tile(np.expand_dims(neighbors, axis=-1), n1).swapaxes(0, 2).swapaxes(1, 2)
    coordinate = idx_expand + neighbors_expand  # (n1, n2, 2)
    # remove those that are not in mask and have no coordinate value
    coordinate[..., 0] = np.where(coordinate[..., 0] >= input_height, input_height - 1, coordinate[..., 0])
    coordinate[..., 0] = np.where(coordinate[..., 0] < 0, 0, coordinate[..., 0])
    coordinate[..., 1] = np.where(coordinate[..., 1] >= input_width, input_width - 1, coordinate[..., 1])
    coordinate[..., 1] = np.where(coordinate[..., 1] < 0, 0, coordinate[..., 1])
    x_val = mask_image_inf[coordinate[..., 0], coordinate[..., 1]][..., 0]  # (n1, n2)
    # set the true to 1, false to 0
    neighbor_mask = np.where((x_val == float('inf')) | (x_val == 0), 0, 1)  # (n1, n2)
    # calculate distance matrix
    D = np.linalg.norm(coordinate - idx_expand, axis=2) + 1e-6  # (n1, n2)
    # calculate the inverse distance weight and update the new matrix
    W = (1 / D) / np.expand_dims(np.sum(1 / D * neighbor_mask, axis=1), axis=1)  # (n1, n2)
    diff_val = np.matmul((neighbor_mask * W).reshape(n1, 1, n2),
                         pred_image[coordinate[..., 0], coordinate[..., 1]]).reshape(n1, 3)
    up_image[idx[..., 0], idx[..., 1]] = diff_val  # (n1, 3) xyz
    return up_image


def up_sample(pred_point, mask_image, intrinsics, radius, interpolation_type, ratio):
    # point: n*3   mask_image: h*w*3
    height, width = mask_image.shape[0], mask_image.shape[1]
    pred_image = map_3d_to_2d(pred_point, intrinsics, height, width)  # (h, w, 3) xyz
    pred_image = np.where(mask_image, pred_image, 0)  # (h, w, 3)
    up_point_image = uneven_upsample_based_mask(pred_image, mask_image, radius)

    tw, th = up_point_image.shape[1] * ratio, up_point_image.shape[0] * ratio
    # INTER_NEAREST  INTER_AREA = INTER_LINEAR
    interpolation = cv2.INTER_NEAREST if interpolation_type == 'nearest' else cv2.INTER_LINEAR
    up_point_image = cv2.resize(up_point_image, (tw, th), interpolation=interpolation)
    mask_image_float = np.where(mask_image, 1., 0.)
    # uses nearest neighbor interpolation, linear interpolation has serrated edges
    mask_image_float = cv2.resize(mask_image_float, (tw, th), interpolation=interpolation)
    up_mask_image_bool = np.where(mask_image_float == 1., True, False)

    intrinsics = intrinsics / 2 if data_type == 'dataset' else intrinsics / 1.5
    point_image = backproject_depth(up_point_image[..., -1] * 1000, intrinsics * ratio)
    # interpolation edge processing: remove edge interference
    point_set = point_image[up_mask_image_bool].reshape(-1, 3)
    return point_set, up_mask_image_bool  # (point_num, 3)


def color_interpolation(color_img, mask_image_bool, interpolation_type, ratio):
    interpolation = cv2.INTER_NEAREST if interpolation_type == 'nearest' else cv2.INTER_LINEAR
    tw, th = color_img.shape[1] * ratio, color_img.shape[0] * ratio
    up_color_image = cv2.resize(color_img, (tw, th), interpolation=interpolation)
    color_set = up_color_image[mask_image_bool].reshape(-1, 3)
    return color_set  # (point_num, 3)


def transform_visual(points):
    matrix = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, -1.0, 0.0],
         [0.0, 0.0, -1.0]]
    )
    points = np.matmul(points, matrix.transpose())
    return points


def reconstruction(image, mask, intrinsics, radius=8, ratio=2, interpolation_type='linear'):
    """
    :param radius: the neighborhood radius of the sample
    :param ratio: sampling multiple
    :param interpolation_type: nearest area linear
    """
    t1 = time.time()
    point = image[..., 9:][mask]
    point_pred = point[np.where(point != 0)].reshape(-1, 3)
    # print('before sample：', len(point_pred))
    point_up_sample, mask_image_bool = up_sample(point_pred, mask, intrinsics, radius, interpolation_type, ratio)
    # print('after sample：', len(point_up_sample))
    t2 = time.time()
    pred_color = color_interpolation(image[..., :3], mask_image_bool, interpolation_type, ratio)
    t3 = time.time()
    print(f'the time of point sample:{t2 - t1}, the time of color sample:{t3 - t2}')
    point_up_sample = transform_visual(point_up_sample)
    return point_up_sample, pred_color


def reconstruct_and_visualize(pre_data, out_path, save_img=True):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960 * 2, height=640 * 2, left=10, top=10)
    pcd = o3d.geometry.PointCloud()
    for i, (image, mask, intrinsics) in enumerate(pre_data):
        point_up_sample, pred_color = reconstruction(image, mask, intrinsics)
        pcd.points = o3d.utility.Vector3dVector(point_up_sample.reshape(-1, 3))
        pcd.colors = o3d.utility.Vector3dVector(pred_color.reshape(-1, 3))
        # visual the single image
        # o3d.visualization.draw_geometries([pcd])
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        if save_img:
            vis.capture_screen_image(f"{out_path}{i:06d}.png")
    vis.destroy_window()


if __name__ == '__main__':
    crop_type = 'inter_nearest'
    input_height, input_width = 240, 320
    obj = 'move_dragon'
    data_type = 'dataset'
    img_num = 90
    data_path = f'/home/PycharmProjects/data/{obj}/'

    pre_data = batch_data_process(img_num, data_path)
    print('data preprocess done!')

    out_path = data_path + '/recon_img/'
    reconstruct_and_visualize(pre_data, out_path)



