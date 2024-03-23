import random
import math
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import cv2


def akaze_registration2d(source_image, target_image, optical_flow_gt=None):
    # running speed and robustness comparison: SIFT < SURF < BRISK < FREAK < ORB < AKAZE
    # initial
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    brisk = cv2.BRISK_create()
    orb = cv2.ORB_create()
    akaze = cv2.AKAZE_create()
    # find key points and descriptions
    kp1, des1 = akaze.detectAndCompute(source_image, None)
    kp2, des2 = akaze.detectAndCompute(target_image, None)

    # BFMatcher (Brute force computing)
    bf = cv2.BFMatcher()   # cv2.NORM_HAMMING, crossCheck=True
    # matches = bf.match(des1, des2)
    matches = bf.knnMatch(des1, des2, k=2)

    """
    # get the flann matcher
    FLANN_INDEX_KDTREE = 0
    # para1：indexParams
    #    for SIFT and SURF: index_params=dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #    for ORB: index_params=dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12)
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # para2：searchParams (the number of recursive traversals)
    searchParams = dict(checks=50)
    # ues FlannBasedMatcher to find the nearest neighbor 
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)
    # use knnMatch and return matches
    matches = flann.knnMatch(des1, des2, k=2)
    """

    # sort by similarity
    matches = sorted(matches, key=lambda x: x[0].distance)
    # rotation test
    good_matches = []
    n = 50
    for d1, d2 in matches:
        # the smaller the coefficient, the fewer the matching points
        if d1.distance < 0.9 * d2.distance:
            good_matches.append([d1])

    # draw matches
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[: n], img2, flags=2)
    img3 = cv2.drawMatchesKnn(np.uint8(source_image), kp1, np.uint8(target_image), kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3.astype('uint8'))
    plt.show()
    cv2.imwrite('akaze_matches.jpg', img3)

    # select matching key
    ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # change the display mode to interactive mode
    plt.ion()
    ndarray_image = np.concatenate([source_image, target_image], axis=1)  # (h, 2w, 3)
    plt.imshow(ndarray_image.astype('uint8'))
    plt.show()
    plt.savefig('akaze_registration2d.png')

    # calculate homography
    H, status = cv2.findHomography(ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC, 5.0)
    # transform
    warped_image = cv2.warpPerspective(source_image, H, (source_image.shape[1], source_image.shape[0]))
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('akaze_warped.jpg', warped_image)

    # calculate error
    if not optical_flow_gt:
        error_matrix = np.zeros((ref_matched_kpts.shape[0],))
        num = 0
        for i in range(ref_matched_kpts.shape[0]):
            u, v = round(ref_matched_kpts[i, 0, 0]), round(ref_matched_kpts[i, 0, 1])
            if np.isinf(optical_flow_gt[v, u, 0]):
                continue
            u_error = abs(optical_flow_gt[v, u, 0]) - abs(ref_matched_kpts[i, 0, 0] - sensed_matched_kpts[i, 0, 0])
            v_error = abs(optical_flow_gt[v, u, 1]) - abs(ref_matched_kpts[i, 0, 1] - sensed_matched_kpts[i, 0, 1])
            error = math.sqrt(u_error ** 2 + v_error ** 2)
            error_matrix[i] = error
            if num < n:
                plt.plot([ref_matched_kpts[i, 0, 0], sensed_matched_kpts[i, 0, 0] + source_image.shape[1]],
                         [ref_matched_kpts[i, 0, 1], sensed_matched_kpts[i, 0, 1]],
                         color=[(10+3*i)/255, (80+i)/255, 220/255], linewidth=0.5, marker='.', markersize=2)
            num += 1
        # remove invalid error
        error_matrix = error_matrix[error_matrix != 0]
        print('akaze EPE2D error: %f pixel ' % np.mean(error_matrix), error_matrix.shape[0])


def icp_registration3d(source_point, target_point, target_point_gt=None):
    source_pcd = o3d.geometry.PointCloud()
    num_point = source_point.shape[0]
    idx = random.sample(range(num_point), num_point)
    source_point = source_point[idx]
    source_pcd.points = o3d.utility.Vector3dVector(source_point)
    source_pcd.paint_uniform_color([255/255, 127/255, 0/255])  # orange
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_point)
    target_pcd.paint_uniform_color([50/255, 205/255, 50/255])  # green

    """
    threshold = 1.0  # the threshold of the moving range
    trans_init = np.asarray([[1, 0, 0, 0],  # 4x4 identity matrix
                             [0, 1, 0, 0],  # Initial matrix: no displacement, no rotation
                             [0, 0, 1, 0],  
                             [0, 0, 0, 1]])
    reg_p2p = o3d.pipelines.registration.registration_icp(source_pcd, target_pcd, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
    source_pcd.transform(reg_p2p.transformation)
    target_pred_pcd = source_pcd
    target_point_pred = np.array(source_pcd.points)
    target_pred_pcd.paint_uniform_color([67 / 255, 110 / 255, 238 / 255])  # blue
    """

    c0 = np.mean(source_point, axis=0)
    c1 = np.mean(target_point, axis=0)
    H = (source_point - c0).transpose() @ (target_point - c1)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    t = c1 - R @ c0
    target_point_pred = np.dot(source_point, R.transpose()) + t
    target_pred_pcd = o3d.geometry.PointCloud()
    target_pred_pcd.points = o3d.utility.Vector3dVector(target_point_pred)
    target_pred_pcd.paint_uniform_color([67/255, 110/255, 238/255])  # blue

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(target_pred_pcd)
    vis.add_geometry(target_pcd)

    """
    # Alignment
    align_colors = [[(10+2*i)%255/255, 80/255, 200/255] for i in range(num_point)]
    icp_points = np.concatenate([target_point_pred, target_point], axis=0)
    icp_lines = [[i, i + num_point] for i in range(num_point)]
    icp_align = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(icp_points),
        lines=o3d.utility.Vector2iVector(icp_lines))
    icp_align.colors = o3d.utility.Vector3dVector(align_colors)
    vis.add_geometry(icp_align)
    """

    # vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()
    vis.run()

    # calculate error
    if not target_point_gt:
        point_error = np.linalg.norm(target_point_pred - target_point_gt, axis=1, ord=2)
        print('ICP EPE3D error: %f m' % np.mean(point_error))


def predict(data):
    source_color = data['source_color']  # (h, w, 3)
    target_color = data['target_color']  # (h, w, 3)
    akaze_registration2d(source_color, target_color)

    # n: number of points
    source_point = data['source_point']         # (n, 3)
    target_point = data['source_target_point']  # (n, 3)
    icp_registration3d(source_point, target_point)
