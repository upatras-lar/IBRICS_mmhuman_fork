import os
import pickle
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from mmhuman3d.core.cameras.camera_parameters import CameraParameter
from mmhuman3d.data.data_converters.builder import DATA_CONVERTERS
from mmhuman3d.data.data_structures.human_data import HumanData
from .base_converter import BaseModeConverter


@DATA_CONVERTERS.register_module()
class Pw3dConverter(BaseModeConverter):
    """3D Poses in the Wild dataset `Recovering Accurate 3D Human Pose in The
    Wild Using IMUs and a Moving Camera' ECCV'2018 More details can be found in
    the `paper.

    <https://virtualhumans.mpi-inf.mpg.de/papers/vonmarcardECCV18/
    vonmarcardECCV18.pdf>`__ .

    Args:
        modes (list): 'test' and/or 'train' for accepted modes
    """

    ACCEPTED_MODES = ['train', 'test']

    def __init__(self, modes: List = []) -> None:
        super(Pw3dConverter, self).__init__(modes)

    def convert_by_mode(self, dataset_path: str, out_path: str,
                        mode: str) -> dict:
        """
        Args:
            dataset_path (str): Path to directory where raw images and
            annotations are stored.
            out_path (str): Path to directory to save preprocessed npz file
            mode (str): Mode in accepted modes

        Returns:
            dict:
                A dict containing keys image_path, bbox_xywh, smpl, meta
                stored in HumanData() format
        """
        # use HumanData to store all data
        human_data = HumanData()
        human_data.set_key_strict(value=True)

        # structs we use
        image_path_, bbox_xywh_ = [], []
        smpl = {
            'body_pose': [],
            'global_orient': [],   # world frame
            'global_trans': [],    # world frame
            'camera_orient': [],   # camera frame
            'camera_trans': [],    # camera frame
            'betas': []
        }

        meta = {}
        # meta['gender'] = []

        # ---- New: numeric camera storage (float32) with deduplicated intrinsics ----
        # Global (dataset-level) intrinsics pool and per-frame index
        K_pool_list: List[np.ndarray] = []
        # key: bytes of K(float32).tobytes(), value: index in K_pool_list
        K_pool_map = {}
        K_idx_list: List[int] = []

        R_list: List[np.ndarray] = []
        T_list: List[np.ndarray] = []
        H_list: List[np.float32] = []
        W_list: List[np.float32] = []
        # ---------------------------------------------------------------------------

        root_path = dataset_path
        # get a list of .pkl files in the directory
        dataset_path = os.path.join(dataset_path, 'sequenceFiles', mode)
        files = [
            os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
            if f.endswith('.pkl')
        ]

        # go through all the .pkl files
        for filename in tqdm(files):
            with open(filename, 'rb') as f:
                data = pickle.load(f, encoding='latin1')

                # used
                seq_name = str(data['sequence'])
                smpl_betas = data['betas']
                smpl_pose = data['poses']
                K = np.array(data['cam_intrinsics'])
                world_trans = data["trans"]
                valid = np.asarray(data['campose_valid'], dtype=bool)

                # maybe
                poses2d = data['poses2d']
                global_poses = data['cam_poses']

                # not in dataset
                num_people = len(smpl_pose)
                num_frames = len(smpl_pose[0])
                img_names = np.array([
                    'imageFiles/' + seq_name + f'/image_{str(i).zfill(5)}.jpg'
                    for i in range(num_frames)
                ])

                # get through all the people in the sequence
                for i in range(num_people):
                    valid_pose = smpl_pose[i][valid[i]]
                    valid_betas = np.tile(smpl_betas[i][:10].reshape(1, -1),
                                          (num_frames, 1))
                    valid_betas = valid_betas[valid[i]]
                    valid_img_names = img_names[valid[i]]
                    valid_global_poses = global_poses[valid[i]]
                    valid_world_trans = world_trans[i][valid[i]]
                    valid_keypoints_2d = poses2d[i][valid[i]] if poses2d is not None else None

                    # consider only valid frames
                    for valid_i in range(valid_pose.shape[0]):
                        have_bbox = False
                        if valid_keypoints_2d is not None:
                            # (18,3): [x,y,conf]
                            k = valid_keypoints_2d[valid_i].T
                            ok = np.isfinite(k[:, 2]) & (k[:, 2] > 0.0)
                            if ok.sum() >= 6:
                                xy = k[ok, :2]
                                x0, y0 = xy.min(axis=0)
                                x1, y1 = xy.max(axis=0)
                                bbox_xyxy = [x0, y0, x1, y1]
                                bbox_xyxy = self._bbox_expand(
                                    bbox_xyxy, scale_factor=1.2)
                                bbox_xywh = self._xyxy2xywh(bbox_xyxy)
                                have_bbox = True
                        if not have_bbox:
                            continue

                        image_path = valid_img_names[valid_i]
                        image_abs_path = os.path.join(root_path, image_path)
                        h, w, _ = cv2.imread(image_abs_path).shape

                        # transform global pose
                        pose = valid_pose[valid_i].copy()
                        l_pose = valid_pose[valid_i].copy()

                        # World Frame

                        smpl['body_pose'].append(pose[3:].reshape((23, 3)))
                        smpl['global_orient'].append(pose[:3])
                        smpl['global_trans'].append(valid_world_trans[valid_i])
                        smpl['betas'].append(valid_betas[valid_i])

                        # meta['gender'].append(gender)

                        # Camera Stuff
                        E = valid_global_poses[valid_i]
                        R = E[:3, :3]
                        T = E[:3, 3]

                        # rotate root orientation into camera frame
                        R_root = cv2.Rodrigues(l_pose[:3])[0]
                        R_root_cam = R @ R_root
                        rvec_cam = cv2.Rodrigues(R_root_cam)[0].ravel()
                        smpl['camera_orient'].append(rvec_cam)

                        # translate root into camera frame
                        t_world = valid_world_trans[valid_i]
                        t_cam = (R @ t_world) + T
                        smpl['camera_trans'].append(t_cam)

                        image_path_.append(image_path)
                        bbox_xywh_.append(bbox_xywh)

                        # REVIEW if necessary
                        # float32 for compact storage
                        K_f32 = np.asarray(K, dtype=np.float32)
                        R_f32 = np.asarray(R, dtype=np.float32)
                        T_f32 = np.asarray(T, dtype=np.float32)

                        # Deduplicate K via a pool
                        k_key = K_f32.tobytes()
                        if k_key in K_pool_map:
                            k_idx = K_pool_map[k_key]
                        else:
                            k_idx = len(K_pool_list)
                            K_pool_map[k_key] = k_idx
                            K_pool_list.append(K_f32)

                        K_idx_list.append(k_idx)
                        R_list.append(R_f32)
                        T_list.append(T_f32)
                        H_list.append(np.float32(h))
                        W_list.append(np.float32(w))

        # change list to np array
        bbox_xywh_ = np.array(bbox_xywh_).reshape((-1, 4))
        bbox_xywh_ = np.hstack([bbox_xywh_, np.ones([bbox_xywh_.shape[0], 1])])
        
        
        
        smpl['global_trans'] = np.asarray(
            smpl['global_trans'],  dtype=np.float32)
        smpl['camera_orient'] = np.asarray(
            smpl['camera_orient'], dtype=np.float32)
        smpl['camera_trans'] = np.asarray(
            smpl['camera_trans'],  dtype=np.float32)
        smpl['body_pose'] = np.asarray(
            smpl['body_pose'],     dtype=np.float32).reshape(-1, 23, 3)
        smpl['global_orient'] = np.asarray(
            smpl['global_orient'], dtype=np.float32).reshape(-1, 3)
        smpl['betas'] = np.asarray(
            smpl['betas'],         dtype=np.float32).reshape(-1, 10)
        
        
        N = len(image_path_)
        assert all(len(smpl[k]) == N for k in ['body_pose','global_orient','global_trans',
                                            'camera_orient','camera_trans','betas'])
        assert len(bbox_xywh_) == N == len(K_idx_list) == len(R_list) == len(T_list) == len(H_list) == len(W_list)
        
        
        # shapes
        assert smpl['body_pose'].shape[1:] == (23,3)
        assert smpl['global_orient'].shape[1] == 3
        assert smpl['global_trans'].shape[1] == 3
        assert smpl['camera_orient'].shape[1] == 3
        assert smpl['camera_trans'].shape[1] == 3

        # meta['gender'] = np.array(meta['gender'])

        human_data['image_path'] = image_path_
        human_data['bbox_xywh'] = bbox_xywh_
        human_data['smpl'] = smpl

        if K_pool_list:
            K_pool = np.stack(K_pool_list, axis=0).astype(
                np.float32, copy=False)  # (M, 3, 3)
        else:
            K_pool = np.empty((0, 3, 3), dtype=np.float32)

        cam_param = {
            'K_pool': K_pool,                                   # unique intrinsics
            # per-frame index
            'K_idx': np.asarray(K_idx_list, dtype=np.int32),
            # (N, 3, 3)
            'R': np.stack(R_list, axis=0).astype(np.float32, copy=False),
            # (N, 3)
            'T': np.stack(T_list, axis=0).astype(np.float32, copy=False),
            'H': np.asarray(H_list, dtype=np.float32),          # (N,)
            'W': np.asarray(W_list, dtype=np.float32),          # (N,)
        }
        human_data['cam_param'] = cam_param
        # -----------------------------------------------------------

        human_data['config'] = 'pw3d'

        # store data
        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        file_name = 'pw3d_{}.npz'.format(mode)
        out_file = os.path.join(out_path, file_name)
        human_data.dump(out_file)
