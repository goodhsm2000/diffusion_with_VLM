from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_backbone import TransfuserBackbone
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from diffusers.schedulers import DDIMScheduler
from navsim.agents.diffusiondrive.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
import torch.nn.functional as F
from navsim.agents.diffusiondrive.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from typing import Any, List, Dict, Optional, Union
import warnings
import cv2
import os
import json
import random
import ray
from ultralytics import YOLO
from navsim.agents.diffusiondrive.eval_traj_vlm import AnchorTrajectoryScorer

# === JSONL 파일에서 action_probs 읽어오는 함수 ===
def load_jsonl_and_extract_action_probs(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    .jsonl 파일에서 각 줄의 JSON 객체를 읽어,
    'token'과 'action_probs'를 파싱하여 dict 형태로 리스트로 반환합니다.
    반환되는 리스트 항목 예시:
      {
        "token": "82521f61cf965167",
        "action_probs": { "forward": 0.5, "veer_left": 0.1, "veer_right": 0.3, "stop": 0.1 }
      }
    """
    results = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[라인 {line_num}] JSON 파싱 오류: {e}")
                continue

            token = obj.get("token", None)
            action_probs = obj.get("action_probs", None)
            if action_probs is None:
                print(f"[라인 {line_num}] 'action_probs' 필드가 없습니다.")
                continue

            # 선택적 검증: 확률 합계가 1에 가까운지 확인
            total_prob = sum(action_probs.values())
            if abs(total_prob - 1.0) > 1e-6:
                print(f"[라인 {line_num}] 확률 합계가 1이 아닙니다: {total_prob}")

            results.append({
                "token": token,
                "action_probs": action_probs
            })
    return results

def world2bev_pixel(xs, ys, min_x, max_y, pixel_size, H_bev, W_bev):
    """
    real-world (x, y) 좌표를 BEV 이미지 픽셀 (u, v) 좌표로 변환
    """
    u = ((xs - min_x) / pixel_size).astype(np.int32)
    v = ((max_y - ys) / pixel_size).astype(np.int32)
    # clamp
    u = np.clip(u, 0, W_bev-1)
    v = np.clip(v, 0, H_bev-1)
    return u, v

class V2TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1,320),
        )

        # visualization directory
        self.save_vis_dir = "/data/goodhsm2000/repos/DiffusionDrive/result"
        os.makedirs(self.save_vis_dir, exist_ok=True)
        self.vis_count = 0

        jsonl_path = "/data/goodhsm2000/repos/InternVL/anchor_probs.jsonl"
        self.token_to_action_probs: Dict[str, Dict[str, float]] = {}
        if jsonl_path is not None:
            data_list = load_jsonl_and_extract_action_probs(jsonl_path)
            for entry in data_list:
                tok = entry["token"]
                ap = entry["action_probs"]
                if tok is not None:
                    self.token_to_action_probs[tok] = ap


    def _visualize(self, bev_sem_map: torch.Tensor, all_anchor_trajs: torch.Tensor):
        """
        bev_sem_map: [C, H, W]
        all_anchor_trajs: [M, T, 3]
        """
        # --- BEV map detach() 추가 ---
        sem = bev_sem_map.detach().cpu().numpy()        # C,H,W
        gray = (sem[0] * 255).astype(np.uint8)
        bev_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        H, W = gray.shape

        # --- 모든 앵커 궤적도 detach() 해 줍니다 ---
        trajs = all_anchor_trajs.detach().cpu().numpy()  # M,T,3
        xs = trajs[..., 0]  # M,T
        ys = trajs[..., 1]  # M,T

        u, v = world2bev_pixel(
            xs, ys,
            self._config.lidar_min_x,
            self._config.lidar_max_y,
            self._config.bev_pixel_size,
            H, W
        )

        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(trajs.shape[0])]
        for m in range(trajs.shape[0]):
            for t in range(trajs.shape[1] - 1):
                cv2.line(
                    bev_img,
                    (int(u[m, t]), int(v[m, t])),
                    (int(u[m, t+1]), int(v[m, t+1])),
                    colors[m], 2
                )

        fname = os.path.join(self.save_vis_dir, f"vis_{self.vis_count}.png")
        cv2.imwrite(fname, bev_img)
        self.vis_count += 1


    def forward(self, features: Dict[str, torch.Tensor], token: str, targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]
        
        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        # upsample to the same shape as bev_feature_upscale

        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        # concat concat_cross_bev and cross_bev_feature
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
        cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1])
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        trajectory = self._trajectory_head(trajectory_query,agents_query, cross_bev_feature,bev_spatial_shape,status_encoding[:, None], self.token_to_action_probs, targets=targets,global_img=None, token=token)
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)

        # === 배치 내 각 샘플에 대해 모든 앵커 궤적 시각화 ===
        # for b in range(batch_size):
        #    self._visualize(
        #        bev_semantic_map[b],          # [C,H,W]
        #        output["all_anchor_trajs"][b] # [M,T,3]
        #    )

        return output


class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape

        # 6. get final prediction
        traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs,ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg, plan_cls
class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
            global_feature = torch.cat([
                    global_img, global_feature
                ], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.time_modulation = ModulationLayer(config.tf_d_model,256)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
        )

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        
        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

        # 4.5 cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        # 4.8 modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed,global_cond=None,global_img=global_img)
        
        # 4.9 predict the offset & heading
        poses_reg, poses_cls = self.task_decoder(traj_feature) #bs,20,8,3; bs,20

        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg, poses_cls
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# anchor 분류를 위한 코드
def classify_anchors(
    anchors: np.ndarray,
    stop_dist_threshold: float = 0.5,
    straight_angle_thresh: float = np.deg2rad(15),
    left_angle_range: tuple = (np.deg2rad(15), np.deg2rad(90)),
    right_angle_range: tuple = (-np.deg2rad(90), -np.deg2rad(15)),
):
    """
    anchors: shape (N, T, 2), 각 앵커의 (x,y) 시퀀스
    stop_dist_threshold: 이동거리가 이 값 이하면 'stop'
    straight_angle_thresh: |angle| < 이 값 이면 'straight'
    left_angle_range: (min_rad, max_rad) 이 범위에 들어오면 'left'
    right_angle_range: (min_rad, max_rad) 이 범위에 들어오면 'right'
    """
    ends = anchors[:, -1, :]  # (N, 2), 마지막 좌표
    dx = ends[:, 0]
    dy = ends[:, 1]
    dist = np.hypot(dx, dy)
    angles = np.arctan2(dy, dx)  # [-pi, pi]

    classes = []
    for d, a in zip(dist, angles):
        if d < stop_dist_threshold:
            classes.append("stop")
        elif -straight_angle_thresh <= a <= straight_angle_thresh:
            classes.append("straight")
        elif left_angle_range[0] <= a <= left_angle_range[1]:
            classes.append("left")
        elif right_angle_range[0] <= a <= right_angle_range[1]:
            classes.append("right")
        else:
            # 이 외의 극단적 꺾임은 straight로 처리
            classes.append("straight")
    return np.array(classes), ends

def extract_anchor_indices(classes: np.ndarray):
    """
    classes: (N,) 문자열 배열 ['straight', 'left', 'right', 'stop']
    반환값:
      dict {
        'straight': [인덱스 리스트],
        'left':     [인덱스 리스트],
        'right':    [인덱스 리스트],
        'stop':     [인덱스 리스트],
      }
    그리고 각 리스트의 개수를 출력함.
    """
    idx_dict = {
        "straight": [],
        "left":     [],
        "right":    [],
        "stop":     [],
    }
    for idx, cls in enumerate(classes):
        if cls in idx_dict:
            idx_dict[cls].append(idx)

    # 개수 출력
    print(f"straight anchors: {len(idx_dict['straight'])}  → {idx_dict['straight']}")
    print(f"left anchors:     {len(idx_dict['left'])}  → {idx_dict['left']}")
    print(f"right anchors:    {len(idx_dict['right'])}  → {idx_dict['right']}")
    print(f"stop anchors:     {len(idx_dict['stop'])}  → {idx_dict['stop']}")

    return idx_dict


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list, poses_cls_list

class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str,config: TransfuserConfig):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )

        # VLM을 이용한 궤적 평가 class 호출
        self.anchor_scorer = AnchorTrajectoryScorer.remote(
            config=config,
            model_name="OpenGVLab/InternVL3-8B",
            hd_map_dir="/data/goodhsm2000/repos/InternVL/HD_Map",
            result_dir="/data/goodhsm2000/result",
        )
        
        # Load a model
        self.model = YOLO("best.pt")  # load an official model

        plan_anchor = np.load(plan_anchor_path)
        # plan_anchor = np.repeat(plan_anchor, 256, axis = 0)


       # 2) 앵커 분류 (여기서 threshold 값은 필요에 따라 변경 가능합니다)
        # 2) 분류 기준 파라미터 설정
        STOP_DIST = 12              # 10m 이내 이동은 stop
        STRAIGHT_THRESH = np.deg2rad(15)   # ±15° 이내는 straight
        LEFT_RANGE = (np.deg2rad(15), np.deg2rad(90))
        RIGHT_RANGE = (-np.deg2rad(90), -np.deg2rad(15))

        classes, _ends = classify_anchors(
            plan_anchor,
            stop_dist_threshold=STOP_DIST,
            straight_angle_thresh=STRAIGHT_THRESH,
            left_angle_range=LEFT_RANGE,
            right_angle_range=RIGHT_RANGE,
        )

        idx_dict = extract_anchor_indices(classes)

        # 3) 자동으로 분류된 인덱스를 멤버 변수에 저장
        self.straight_anchor_idxs = idx_dict.get("straight", [])
        self.left_anchor_idxs     = idx_dict.get("left", [])
        self.right_anchor_idxs    = idx_dict.get("right", [])
        self.stop_anchor_idxs     = idx_dict.get("stop", [])


        # self.straight_anchor_idxs = [1, 3, 4, 6, 11, 12, 14, 15, 17, 18]    # 예시: 앞으로 주로 쓰는 앵커 인덱스
        # self.left_anchor_idxs     = [0, 5, 7, 13, 16]   # 예시: 왼쪽으로 턴하는 앵커 인덱스
        # self.right_anchor_idxs    = [9, 10, 19]  # 예시: 오른쪽으로 턴하는 앵커 인덱스
        # self.stop_anchor_idxs     = [2, 8]  # 예시: 정지 앵커 인덱스

        # 여기서 sampling된 plane_anchor 설정
    
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) # 20,8,2
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1,512),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)

        self.loss_computer = LossComputer(config)
    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2*(odo_info_fut_x + 1.2)/56.9 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 20)/46 -1
        odo_info_fut_head = 2*(odo_info_fut_head + 2)/3.9 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * 46 - 20
        odo_info_fut_head = (odo_info_fut_head + 1)/2 * 3.9 - 2
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    def forward(self, ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding, token_to_action_probs, targets=None,global_img=None, token=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.training:
            return self.forward_train(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,targets,global_img)
        else:
            return self.forward_test(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding, token_to_action_probs, global_img, token)


    def forward_train(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None) -> Dict[str, torch.Tensor]:
        bs = ego_query.shape[0]
        device = ego_query.device
        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)
        odo_info_fut = self.norm_odo(plan_anchor)
        # 1000step 중 앞 5%에 노이즈 주입
        timesteps = torch.randint(
            0, 50,
            (bs,), device=device
        )
        noise = torch.randn(odo_info_fut.shape, device=device)
        noisy_traj_points = self.diffusion_scheduler.add_noise(
            original_samples=odo_info_fut,
            noise=noise,
            timesteps=timesteps,
        ).float()
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        noisy_traj_points = self.denorm_odo(noisy_traj_points)
        
        ego_fut_mode = noisy_traj_points.shape[1]
        # 2. proj noisy_traj_points to the query
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs,ego_fut_mode,-1)
        # 3. embed the timesteps
        time_embed = self.time_mlp(timesteps)
        time_embed = time_embed.view(bs,1,-1)


        # 4. begin the stacked decoder
        poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)

        trajectory_loss_dict = {}
        ret_traj_loss = 0
        for idx, (poses_reg, poses_cls) in enumerate(zip(poses_reg_list, poses_cls_list)):
            trajectory_loss = self.loss_computer(poses_reg, poses_cls, targets, plan_anchor)
            trajectory_loss_dict[f"trajectory_loss_{idx}"] = trajectory_loss
            ret_traj_loss += trajectory_loss

        # assert 1 == 0, (
        #     f"[ERROR] Invalid plan_anchor: expected != 20 trajectories, but got {len(poses_reg_list), len(poses_cls_list)}"
        # )

        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        warnings.warn(f"[DEBUG] chosen_anchor_idx = {mode_idx.tolist()}")
        mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)

        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)
        warnings.warn(f"trajectory_loss_dict = { {k: v.item() for k,v in trajectory_loss_dict.items()} }")
        return {"trajectory": best_reg,"trajectory_loss":ret_traj_loss,"trajectory_loss_dict":trajectory_loss_dict, "all_anchor_trajs": poses_reg_list[-1],
                "all_anchor_scores": poses_cls_list[-1]}

    def forward_test(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding, token_to_action_probs, global_img, token=None) -> Dict[str, torch.Tensor]:
        """
        1) 원본 코드의 구조(차원, layer 호출 등)를 최대한 그대로 유지합니다.
        2) 배치별로 N개의 앵커를 token_to_action_probs를 참조하여 샘플링하는 로직만 추가되어 있습니다.
        """

        # ------------------------ 0. diffusion scheduler 세팅 (원본과 완전 동일) ------------------------
        step_num = 2
        bs = ego_query.shape[0]
        device = ego_query.device
        self.diffusion_scheduler.set_timesteps(1000, device=device)
        step_ratio = 20 / step_num
        roll_timesteps = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
        roll_timesteps = torch.from_numpy(roll_timesteps).to(device)
        warnings.warn(f"bs = {bs}")
        warnings.warn(f"token = {token}")

        # ------------------------ 1. 원본 앵커(self.plan_anchor)를 bs에 맞춰 복제 → [bs, M, T, D] ------------------------
        #    이때 M,T,D는 self.plan_anchor의 shape[1:], 예: [20, traj_len, 3] 등
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)  # [bs, M, T, D]
        M, T, D = plan_anchor.shape[1:]  # e.g. M=20, T=traj_len, D=3

        # 1-1) 행동(action)별 pre-defined 앵커 인덱스
        straight_anchor_idxs = self.straight_anchor_idxs  # 예: [0,1,2,...]
        left_anchor_idxs     = self.left_anchor_idxs      # 예: [3,4,...]
        right_anchor_idxs    = self.right_anchor_idxs     # 예: [...]
        stop_anchor_idxs     = self.stop_anchor_idxs      # 예: [...]

        # 1-2) 배치마다 N개의 앵커를 token_to_action_probs에 따라 샘플링
        # N = 5
        # anchor_idx = torch.zeros((bs, N), dtype=torch.long, device=device)

        # for b in range(bs):
        #     # 1-2-a) 배치 b의 token에 맞는 action_probs 가져오기
        #     cur_token = token
        #     entry = token_to_action_probs.get(cur_token)
        #     if entry is None:
        #         raise KeyError(f"Token {cur_token} not found in token_to_action_probs.")
        #     probs = entry  # 예: {"forward":0.5, "veer_left":0.2, ...}

        #     # 1-2-b) 각 action별로 샘플 개수 계산
        #     n_forward = int(probs.get("forward", 0.0) * N)
        #     n_left    = int(probs.get("veer_left", 0.0) * N)
        #     n_right   = int(probs.get("veer_right", 0.0) * N)
        #     n_stop    = int(probs.get("stop", 0.0) * N)

        #     total_assigned = n_forward + n_left + n_right + n_stop
        #     if total_assigned != N:
        #         # 반올림 오차로 N보다 작으면 모자란 개수를 forward에 추가
        #         n_forward += (N - total_assigned)

        #     picks = []
        #     picks += random.choices(straight_anchor_idxs, k=n_forward)
        #     picks += random.choices(left_anchor_idxs,     k=n_left)
        #     picks += random.choices(right_anchor_idxs,    k=n_right)
        #     picks += random.choices(stop_anchor_idxs,     k=n_stop)
        #     # random.shuffle(picks)
        #     anchor_idx[b] = torch.tensor(picks, dtype=torch.long, device=device)

             # ------------------------ 1-2) 배치마다 N개의 앵커를 token_to_action_probs에 따라 샘플링 ------------------------
        N = 20
        anchor_idx = torch.zeros((bs, N), dtype=torch.long, device=device)

        for b in range(bs):
            # 1-2-a) 배치 b의 token에 맞는 action_probs 가져오기
            # Predict with the model
            image_path = f"/local_datasets/VLM_input/{token}.jpg"
            results = self.model(image_path) 
            if results is None:
                raise KeyError(f"Token {token} not found in token_to_action_probs.")
            probs = results  # 예: YOLO 각 Class의 결과 

            # 1-2-b) 각 action별로 샘플 개수 계산
            n_forward = int(probs[0] * N)
            n_left    = int(probs[1] * N)
            n_right   = int(probs[2] * N)
            n_stop    = int(probs[3] * N)

            total_assigned = n_forward + n_left + n_right + n_stop
            if total_assigned != N:
                # 반올림 오차로 N보다 작으면 모자란 개수를 forward에 추가
                n_forward += (N - total_assigned)

            picks = []

            # --- forward(직진) 카테고리 ---
            k = n_forward
            avail = straight_anchor_idxs
            if k <= len(avail):
                # 중복 없이 k개 뽑기
                picks += random.sample(avail, k)
            else:
                # avail 전부 뽑고, 나머지는 중복 허용
                picks += random.sample(avail, len(avail))
                picks += random.choices(avail, k=k - len(avail))

            # --- veer_left(좌회전) 카테고리 ---
            k = n_left
            avail = left_anchor_idxs
            if k <= len(avail):
                picks += random.sample(avail, k)
            else:
                picks += random.sample(avail, len(avail))
                picks += random.choices(avail, k=k - len(avail))

            # --- veer_right(우회전) 카테고리 ---
            k = n_right
            avail = right_anchor_idxs
            if k <= len(avail):
                picks += random.sample(avail, k)
            else:
                picks += random.sample(avail, len(avail))
                picks += random.choices(avail, k=k - len(avail))

            # --- stop(정지) 카테고리 ---
            k = n_stop
            avail = stop_anchor_idxs
            if k <= len(avail):
                picks += random.sample(avail, k)
            else:
                picks += random.sample(avail, len(avail))
                picks += random.choices(avail, k=k - len(avail))

            # 이제 picks에는 정확히 N개의 앵커 인덱스가 들어 있습니다.
            # (각 카테고리별로 중복 없이 최대한 뽑고, 부족한 부분만 중복 허용으로 채웠음)
            random.shuffle(picks)  # 순서를 무작위로 섞고 싶다면

            anchor_idx[b] = torch.tensor(picks, dtype=torch.long, device=device)


        # 1-3) [bs, M, T, D] → [bs, N, T, D]로 gather
        #   - plan_anchor         : [bs, M, T, D]
        #   - input_expand        : [bs, N, M, T, D]
        #   - idx_expand          : [bs, N, 1, 1, 1] → [bs, N, 1, T, D]
        #   - torch.gather(..., dim=2) → [bs, N, T, D]
        input_expand = plan_anchor.unsqueeze(1).expand(bs, N, M, T, D)  # [bs, N, M, T, D]
        idx_expand = anchor_idx.view(bs, N, 1, 1, 1).expand(bs, N, 1, T, D)  # [bs, N, 1, T, D]
        sampled = torch.gather(input_expand, dim=2, index=idx_expand).squeeze(2)  # [bs, N, T, D]
        warnings.warn(f"anchor = {picks}")

        # 1-4) flatten → [bs*N, T, D]
        # sampled_flat = sampled.view(bs * N, T, D)  # [bs*N, T, D]

        # ------------------------ 2. Flatten된 앵커 샘플에 노이즈 추가 (원본과 동일) ------------------------
        img = self.norm_odo(sampled)
        noise = torch.randn(img.shape, device=device)
        trunc_timesteps = torch.ones((bs,), device=device, dtype=torch.long) * 8
        img = self.diffusion_scheduler.add_noise(original_samples=img, noise=noise, timesteps=trunc_timesteps)
        noisy_trajs = self.denorm_odo(img)
        ego_fut_mode = img.shape[1]
        for k in roll_timesteps[:]:
            x_boxes = torch.clamp(img, min=-1, max=1)
            noisy_traj_points = self.denorm_odo(x_boxes)

            # 2. proj noisy_traj_points to the query
            traj_pos_embed = gen_sineembed_for_position(noisy_traj_points,hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs,ego_fut_mode,-1)

            timesteps = k
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=img.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(img.device)
            
            # 3. embed the timesteps
            timesteps = timesteps.expand(img.shape[0])
            time_embed = self.time_mlp(timesteps)
            time_embed = time_embed.view(bs,1,-1)

            # 4. begin the stacked decoder
            poses_reg_list, poses_cls_list = self.diff_decoder(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg = poses_reg_list[-1]
            poses_cls = poses_cls_list[-1]
            x_start = poses_reg[...,:2]
            x_start = self.norm_odo(x_start)
            img = self.diffusion_scheduler.step(
                model_output=x_start,
                timestep=k,
                sample=img
            ).prev_sample
        
        # 기존 코드
        # mode_idx = poses_cls.argmax(dim=-1)
        # warnings.warn(f"[DEBUG] chosen_anchor_idx = {mode_idx.tolist()}")
        # mode_idx = mode_idx[...,None,None,None].repeat(1,1,self._num_poses,3)

        # 1) 상위 K=5개 인덱스 추출
        topk_vals, topk_idxs = torch.topk(poses_cls, k=5, dim=1)  # both [bs,5]
        warnings.warn(f"topk_idxs = {topk_idxs}")

        # 2) 배치별로 AnchorTrajectoryScorer에 token과 앵커 배열 전달
        final_trajs = []
        for b in range(bs):
            # a) 토큰 문자열
            cur_token = token

            # b) torch → numpy 변환: (5, T, D)
            selected_np = poses_reg[b, topk_idxs[b]].detach().cpu().numpy()

            # c) VLM-based 정밀 스코어링 & 최종 궤적 선택
            best_np = self.anchor_scorer.select_best_anchor.remote(
                token    = cur_token,
                anchors  = selected_np
            )  # returns np.ndarray shape (1, T, D)
            best_np = ray.get(best_np) 
            # d) torch.Tensor로 변환
            best_t = torch.from_numpy(best_np).to(device).squeeze(0)  # (T, D)
            final_trajs.append(best_t)

        # 3) 배치 차원 복원 → [bs, T, D]
        best_reg = torch.stack(final_trajs, dim=0)

        # 최종 반환
        return {
            "trajectory":        best_reg,         # VLM으로 재선택된 궤적
            "all_anchor_trajs":  poses_reg,
            "all_anchor_scores": poses_cls,
        }


        # best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        # return {"trajectory": best_reg, "all_anchor_trajs": poses_reg_list[-1],
        #         "all_anchor_scores": poses_cls_list[-1]}

