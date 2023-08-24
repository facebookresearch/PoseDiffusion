from collections import defaultdict
import torch
import pytorch3d
import numpy as np
import copy
from typing import Dict, List, Optional, Union, Tuple
import torch
import theseus as th

from util.triangulation import intersect_skew_line_groups
from torchlie.functional import SE3 as SE3_base


class ReprojectionBatch(th.CostFunction):
    def __init__(
        self,
        camera_pose: th.SE3,
        world_point: th.Point3,
        image_feature_point: th.Point2,
        focal_length: th.Vector,
        calib_k1: th.Vector = None,
        calib_k2: th.Vector = None,
        weight: Optional[th.CostWeight] = None,
        name: Optional[str] = None,
        camera_idx = None,
        point_idx = None,
        device = None,
    ):
        if weight is None:
            weight = th.ScaleCostWeight(torch.tensor(1.0).to(dtype=camera_pose.dtype))
            weight.to(device)
        super().__init__(
            cost_weight=weight,
            name=name,
        )
        self.camera_pose = camera_pose
        self.focal_length = focal_length
        self.calib_k1 = calib_k1
        self.calib_k2 = calib_k2
        self.camera_idx = camera_idx
        self.point_idx = point_idx
            
        self.world_point = world_point
        self.image_feature_point = image_feature_point

        self.register_optim_vars(["camera_pose", "world_point"])
        self.register_aux_vars(
            ["focal_length", "image_feature_point", "point_idx", "camera_idx"]
        )

    def error(self) -> torch.Tensor:
        point_all = self.world_point.tensor.reshape(-1,3) 
        pose_all = self.camera_pose.tensor.reshape(-1,3,4)
        fl_all = self.focal_length.tensor.reshape(-1,1)
        
        point2d_all = self.image_feature_point.tensor
        camera_idx_all = self.camera_idx.tensor.long()
        point_idx_all = self.point_idx.tensor.long()
        cameras_compute = pose_all[camera_idx_all[:,0]]
        fl_compute = fl_all[camera_idx_all[:,0]]
        point_compute = point_all[point_idx_all[:,0]]
        
        point_cam = SE3_base.transform(cameras_compute, point_compute)

        proj = -point_cam[:, :2] / point_cam[:, 2:3]
        proj_sqn = (proj * proj).sum(dim=1).unsqueeze(1)
        proj_factor = fl_compute * (1.0 + proj_sqn * (0 + proj_sqn * 0))

        point_projection = proj * proj_factor
        err = point_projection - point2d_all
        return err

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        point_all = self.world_point.tensor.reshape(-1,3) 
        pose_all = self.camera_pose.tensor.reshape(-1,3,4)
        fl_all = self.focal_length.tensor.reshape(-1,1)
        
        point2d_all = self.image_feature_point.tensor
        camera_idx_all = self.camera_idx.tensor.long()
        point_idx_all = self.point_idx.tensor.long()
        cameras_compute = pose_all[camera_idx_all[:,0]]
        fl_compute = fl_all[camera_idx_all[:,0]]
        point_compute = point_all[point_idx_all[:,0]]
                
        cpose_wpt_jacs: List[torch.Tensor] = []

        point_cam = SE3_base.transform(cameras_compute, point_compute, jacobians=cpose_wpt_jacs)
        J = torch.cat(cpose_wpt_jacs, dim=2)

        proj = -point_cam[:, :2] / point_cam[:, 2:3]
        proj_sqn = (proj * proj).sum(dim=1).unsqueeze(1)
        proj_factor = fl_compute * (1.0 + proj_sqn * (0 + proj_sqn * 0))
        
        d_proj_factor = fl_compute * (0 + 2.0 * proj_sqn * 0)
        point_projection = proj * proj_factor

        # derivative of N/D is (N' - ND'/D) / D
        d_num = J[:, 0:2, :]
        num_dden_den = torch.bmm(
            point_cam[:, :2].unsqueeze(2),
            (J[:, 2, :] / point_cam[:, 2:3]).unsqueeze(1),
        )
        proj_jac = (num_dden_den - d_num) / point_cam[:, 2:].unsqueeze(2)
        proj_sqn_jac = 2.0 * proj.unsqueeze(2) * torch.bmm(proj.unsqueeze(1), proj_jac)
        point_projection_jac = proj_jac * proj_factor.unsqueeze(
            2
        ) + proj_sqn_jac * d_proj_factor.unsqueeze(2)

        err = point_projection - point2d_all
        return [point_projection_jac[..., :6], point_projection_jac[..., 6:]], err


    def dim(self) -> int:
        return 2

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

    def _copy_impl(self, new_name: Optional[str] = None) -> "ReprojectionBatch":
        return ReprojectionBatch(
            self.camera_pose.copy(),
            self.world_point.copy(),
            self.image_feature_point.copy(),
            self.focal_length.copy(),
            weight=self.weight.copy(),
            camera_idx = self.camera_idx.copy(),
            point_idx = self.point_idx.copy(),
            name=new_name,
        )


class ReprojectionTH(th.CostFunction):
    def __init__(
        self,
        camera_pose: th.SE3,
        world_point: th.Point3,
        image_feature_point: th.Point2,
        focal_length: th.Vector,
        calib_k1: th.Vector = None,
        calib_k2: th.Vector = None,
        weight: Optional[th.CostWeight] = None,
        name: Optional[str] = None,
        device = None,
    ):
        if weight is None:
            weight = th.ScaleCostWeight(torch.tensor(1.0).to(dtype=camera_pose.dtype))
            weight.to(device)
        super().__init__(
            cost_weight=weight,
            name=name,
        )
        self.camera_pose = camera_pose
        self.focal_length = focal_length
        self.calib_k1 = calib_k1
        self.calib_k2 = calib_k2
        batch_size = self.camera_pose.shape[0]
        if self.calib_k1 is None:
            self.calib_k1 = th.Vector(
                tensor=torch.zeros((batch_size, 1), dtype=camera_pose.dtype),
                name=  "calib_k1",
            )
            self.calib_k1.to(device)
            
        if self.calib_k2 is None:
            self.calib_k2 = th.Vector(
                tensor=torch.zeros((batch_size, 1), dtype=camera_pose.dtype),
                name=  "calib_k2",
            )
            self.calib_k2.to(device)
            
        self.world_point = world_point
        self.image_feature_point = image_feature_point

        self.register_optim_vars(["camera_pose", "world_point"])
        self.register_aux_vars(
            ["focal_length", "image_feature_point", "calib_k1", "calib_k2"]
        )

    def error(self) -> torch.Tensor:
        point_cam = self.camera_pose.transform_from(self.world_point)
        proj = -point_cam[:, :2] / point_cam[:, 2:3]
        proj_sqn = (proj * proj).sum(dim=1).unsqueeze(1)
        proj_factor = self.focal_length.tensor * (
            1.0 + proj_sqn * (self.calib_k1.tensor + proj_sqn * self.calib_k2.tensor)
        )
        point_projection = proj * proj_factor

        err = point_projection - self.image_feature_point.tensor
        if err.shape[0]>1:
            # err.shape: torch.Size([10000, 2])
            # (Pdb) err.max()
            # tensor(0.4748, device='cuda:0', dtype=torch.float64)
            # (Pdb) err.std()
            # tensor(0.0561, device='cuda:0', dtype=torch.float64)

            print(torch.norm(err, dim=1).mean())
        return err

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        cpose_wpt_jacs: List[torch.Tensor] = []

        point_cam = self.camera_pose.transform_from(self.world_point, cpose_wpt_jacs)
        J = torch.cat(cpose_wpt_jacs, dim=2)

        proj = -point_cam[:, :2] / point_cam[:, 2:3]
        proj_sqn = (proj * proj).sum(dim=1).unsqueeze(1)
        proj_factor = self.focal_length.tensor * (
            1.0 + proj_sqn * (self.calib_k1.tensor + proj_sqn * self.calib_k2.tensor)
        )
        d_proj_factor = self.focal_length.tensor * (
            self.calib_k1.tensor + 2.0 * proj_sqn * self.calib_k2.tensor
        )
        point_projection = proj * proj_factor

        # derivative of N/D is (N' - ND'/D) / D
        d_num = J[:, 0:2, :]
        num_dden_den = torch.bmm(
            point_cam[:, :2].unsqueeze(2),
            (J[:, 2, :] / point_cam[:, 2:3]).unsqueeze(1),
        )
        proj_jac = (num_dden_den - d_num) / point_cam[:, 2:].unsqueeze(2)
        proj_sqn_jac = 2.0 * proj.unsqueeze(2) * torch.bmm(proj.unsqueeze(1), proj_jac)
        point_projection_jac = proj_jac * proj_factor.unsqueeze(
            2
        ) + proj_sqn_jac * d_proj_factor.unsqueeze(2)

        err = point_projection - self.image_feature_point.tensor

        return [point_projection_jac[..., :6], point_projection_jac[..., 6:]], err

    def dim(self) -> int:
        return 2

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

    def _copy_impl(self, new_name: Optional[str] = None) -> "ReprojectionTH":
        return ReprojectionTH(
            self.camera_pose.copy(),
            self.world_point.copy(),
            self.image_feature_point.copy(),
            self.focal_length.copy(),
            calib_k1=self.calib_k1.copy(),
            calib_k2=self.calib_k2.copy(),
            weight=self.weight.copy(),
            name=new_name,
        )

class ReprojectionPT3D(th.CostFunction):
    def __init__(
        self,
        camera_pose: th.SE3,
        world_point: th.Point3,
        image_feature_point: th.Point2,
        focal_length: th.Vector,
        calib_k1: th.Vector = None,
        calib_k2: th.Vector = None,
        weight: Optional[th.CostWeight] = None,
        name: Optional[str] = None,
        device = None,
    ):
        if weight is None:
            weight = th.ScaleCostWeight(torch.tensor(1.0).to(dtype=camera_pose.dtype))
            weight.to(device)
        super().__init__(
            cost_weight=weight,
            name=name,
        )
        self.camera_pose = camera_pose
        self.focal_length = focal_length
        self.calib_k1 = calib_k1
        self.calib_k2 = calib_k2
        batch_size = self.camera_pose.shape[0]
        if self.calib_k1 is None:
            self.calib_k1 = th.Vector(
                tensor=torch.zeros((batch_size, 1), dtype=camera_pose.dtype),
                name="calib_k1",
            )
            self.calib_k1.to(device)
        if self.calib_k2 is None:
            self.calib_k2 = th.Vector(
                tensor=torch.zeros((batch_size, 1), dtype=camera_pose.dtype),
                name="calib_k2",
            )
            self.calib_k2.to(device)
            
        self.world_point = world_point
        self.image_feature_point = image_feature_point

        self.register_optim_vars(["camera_pose", "world_point"])
        self.register_aux_vars(
            ["focal_length", "image_feature_point", "calib_k1", "calib_k2"]
        )

    def error(self) -> torch.Tensor:
        point_cam = self.camera_pose.transform_from(self.world_point)
        
        # HEEEEERE!
        # proj = -point_cam[:, :2] / point_cam[:, 2:3]
        proj = point_cam[:, :2] / point_cam[:, 2:3]
        
        proj_sqn = (proj * proj).sum(dim=1).unsqueeze(1)
        proj_factor = self.focal_length.tensor * (
            1.0 + proj_sqn * (self.calib_k1.tensor + proj_sqn * self.calib_k2.tensor)
        )
        point_projection = proj * proj_factor

        err = point_projection - self.image_feature_point.tensor
        return err

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        cpose_wpt_jacs: List[torch.Tensor] = []
        point_cam = self.camera_pose.transform_from(self.world_point, cpose_wpt_jacs)
        J = torch.cat(cpose_wpt_jacs, dim=2)

        # HEEEEERE!
        # proj = -point_cam[:, :2] / point_cam[:, 2:3]
        proj = point_cam[:, :2] / point_cam[:, 2:3]
        
        proj_sqn = (proj * proj).sum(dim=1).unsqueeze(1)
        proj_factor = self.focal_length.tensor * (
            1.0 + proj_sqn * (self.calib_k1.tensor + proj_sqn * self.calib_k2.tensor)
        )
        d_proj_factor = self.focal_length.tensor * (
            self.calib_k1.tensor + 2.0 * proj_sqn * self.calib_k2.tensor
        )
        point_projection = proj * proj_factor

        # derivative of N/D is (N' - ND'/D) / D
        d_num = J[:, 0:2, :]
        num_dden_den = torch.bmm(
            point_cam[:, :2].unsqueeze(2),
            (J[:, 2, :] / point_cam[:, 2:3]).unsqueeze(1),
        )
        # HEEEEERE!
        # proj_jac = (num_dden_den - d_num) / point_cam[:, 2:].unsqueeze(2)
        proj_jac = (num_dden_den - d_num) / point_cam[:, 2:].unsqueeze(2)

        proj_sqn_jac = 2.0 * proj.unsqueeze(2) * torch.bmm(proj.unsqueeze(1), proj_jac)
        point_projection_jac = proj_jac * proj_factor.unsqueeze(
            2
        ) + proj_sqn_jac * d_proj_factor.unsqueeze(2)

        err = point_projection - self.image_feature_point.tensor
        return [point_projection_jac[..., :6], point_projection_jac[..., 6:]], err

    def dim(self) -> int:
        return 2

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

    def _copy_impl(self, new_name: Optional[str] = None) -> "ReprojectionPT3D":
        return ReprojectionPT3D(
            self.camera_pose.copy(),
            self.world_point.copy(),
            self.image_feature_point.copy(),
            self.focal_length.copy(),
            calib_k1=self.calib_k1.copy(),
            calib_k2=self.calib_k2.copy(),
            weight=self.weight.copy(),
            name=new_name,
        )


def process_node_once(track_num, nodes, graph, keypoints, pred_cameras, one_vec, device, image_info, viz, cfg):
    p_list = []; r_list = []

    for node in nodes:
        image_id = node.image_id
        feature_idx = node.feature_idx
        image_name  = graph.image_id_to_name[image_id]
        image_kp = keypoints[image_name]
        kp = torch.from_numpy(image_kp[feature_idx]).view(1,1,-1)
        cam = pred_cameras[image_id]
        
        xy_pair = torch.cat([torch.cat([kp, i*one_vec], -1).to(device).float() for i in [1,10]],dim=0)
        xyz_pair = cam.unproject_points(xy_pair, world_coordinates=True, from_ndc=True)
        xyz_world_1 = xyz_pair[0:1]
        direction = xyz_pair[0:1] - xyz_pair[1:2]

        direction = torch.nn.functional.normalize(direction, dim=-1)
        p_list.append(xyz_world_1)
        r_list.append(direction)
        

    p_list = torch.cat(p_list,dim=1).unsqueeze(0)
    r_list = torch.cat(r_list,dim=1).unsqueeze(0)
    p_intersect, p_line_intersect, inter_dis = intersect_skew_line_groups(p_list, r_list, None)

    if (inter_dis > cfg.match.line_dis_thres).any():
        mask = inter_dis < cfg.match.line_dis_thres
        mask = mask.view(-1)
        mask_list = mask.tolist()
        selected_nodes = [node for node, m in zip(nodes, mask_list) if m]
        
        if len(selected_nodes) > cfg.match.track_len_min:
            return p_intersect, p_line_intersect, inter_dis, selected_nodes         
        
    return p_intersect, p_line_intersect, inter_dis, None

def process_node(track_num, nodes, graph, keypoints, 
                 pred_cameras, one_vec, device, image_info,
                 viz, cfg=None):

    if len(nodes) > cfg.match.track_len_min:
        #
        p_intersect, p_line_intersect, inter_dis, selected_nodes = process_node_once(track_num, nodes, graph, keypoints, 
                                                pred_cameras, one_vec, device, image_info, viz, cfg)
        
        #
        if selected_nodes is not None:
            p_intersect, p_line_intersect, inter_dis, _ = process_node_once(track_num, selected_nodes, 
                                        graph, keypoints, pred_cameras, one_vec, device, image_info, viz, cfg)
        
        for node in nodes:
            image_id = node.image_id
            feature_idx = node.feature_idx
            image_name  = graph.image_id_to_name[image_id]
            image_kp = keypoints[image_name]
            kp = torch.from_numpy(image_kp[feature_idx]).view(1,1,-1)
            cam = pred_cameras[image_id]
            
            proj_pt = cam.transform_points(p_intersect)[:, :, :2]
            node.pt = np.array(kp.cpu())
            node.proj_pt = np.array(proj_pt.cpu())
            node.pt3d = np.array(p_intersect.cpu())
    
        return track_num, p_intersect
    else:
        return track_num, None


def gather_nodes_by_track(graph, track_labels):
    # Create a dictionary where the keys are the track labels and the values are lists of nodes.
    track_to_nodes = defaultdict(list)
    for node_idx, track_label in enumerate(track_labels):
        graph.nodes[node_idx].track_idx = track_label
        # Append the node to the list of nodes for its track.
        track_to_nodes[track_label].append(graph.nodes[node_idx])
    return track_to_nodes



def colmap_kpdict_to_pytorch3d(kps, image_info, namemap):
    keypoints = copy.deepcopy(kps)
    kp1, kp2, i12 = [], [], []
    bbox_xyxy, scale = image_info["bboxes_xyxy"], image_info["resized_scales"]

    for name in keypoints:
        # coordinate change from COLMAP to OpenCV
        idx = namemap[name]

        hw = image_info["rawsize"]
        # TODO: we don't need this anymore because now we read from hloc
        cur_keypoint = keypoints[name]
        
        new_keypoint = convert_to_ndc(cur_keypoint, hw)
        
        keypoints[name] = new_keypoint
    return keypoints




def convert_to_ndc(kps, hw):
    keypoints = copy.deepcopy(kps)
    # Image shape
    h, w = hw

    # Calculate aspect ratio
    aspect_ratio = max(h, w) / min(h, w)

    # Convert to NDC
    # For x-coordinates
    keypoints[:, 0] = 1. - 2. * keypoints[:, 0] / (w - 1.)
    # For y-coordinates
    keypoints[:, 1] = 1. - 2. * keypoints[:, 1] / (h - 1.)

    # Adjust for aspect ratio
    if w > h:
        keypoints[:, 0] *= aspect_ratio
    else:
        keypoints[:, 1] *= aspect_ratio

    return keypoints


def convert_from_ndc(kps, hw):
    keypoints = copy.deepcopy(kps)
    # Image shape
    h, w = hw

    # Calculate aspect ratio
    aspect_ratio = max(h, w) / min(h, w)

    # Adjust for aspect ratio
    if w > h:
        keypoints[:, 0] /= aspect_ratio
    else:
        keypoints[:, 1] /= aspect_ratio

    # Convert from NDC
    # For x-coordinates
    keypoints[:, 0] = (1. - keypoints[:, 0]) * (w - 1.) / 2.
    # For y-coordinates
    keypoints[:, 1] = (1. - keypoints[:, 1]) * (h - 1.) / 2.

    return keypoints







