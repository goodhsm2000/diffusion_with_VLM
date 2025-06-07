#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnchorTrajectoryScorer (최종 버전)
— HD‐Map BEV 이미지는 ego‐centric이며, (0,0)이 하단 중앙에 대응합니다.
— 각 앵커 궤적(local: [N, 8, 2] in meters, (0,0) == 하단 중앙) 을 차례로
  빨간 선으로 오버레이하여 VLM에 전달하고 score를 얻습니다.
— 모든 앵커를 한 장의 HD‐Map 위에 오버레이한 이미지, 각 앵커별 오버레이 이미지,
  그리고 최종 선택된 앵커를 오버레이한 이미지를 저장합니다.
— 모든 앵커의 score는 JSONL으로 (anchor_idx, score) 형태로 기록합니다.
"""

import json
import os
from pathlib import Path
from typing import Optional

import traceback
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import AutoModel, AutoTokenizer
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig

print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime version:", torch.version.cuda)
print("GPU name:", torch.cuda.get_device_name(0))

def pad_to_square(img: Image.Image, fill=(0,0,0)):
    w, h = img.size
    if w == h: return img
    diff = abs(w-h)//2
    padding = (0, diff, 0, w-h-diff) if w>h else (diff, 0, w-h-diff, 0)
    return ImageOps.expand(img, padding, fill=fill)

# -------------------------------------------------------------------
# 1) AnchorTrajectoryScorer 클래스 정의
# -------------------------------------------------------------------
class AnchorTrajectoryScorer:
    def __init__(
        self,
        config: TransfuserConfig,  # TransfuserConfig 인스턴스 (bev_pixel_size, bev_pixel_width, bev_pixel_height 정보 포함)
        model_name: str = "OpenGVLab/InternVL2_5-4B",
        hd_map_dir: str = "/data/goodhsm2000/repos/InternVL/HD_Map",
        result_dir: str = "/data/goodhsm2000/result",
    ):
        """
        Args:
            config: TransfuserConfig 인스턴스
            model_name: InternVL3 checkpoint 이름 (HuggingFace)
            hd_map_dir: "{token}_hd.png" 파일들이 있는 디렉토리
            result_dir: 앵커 오버레이 결과 및 JSONL을 저장할 디렉토리
        """
        self.hd_map_dir = Path(hd_map_dir)
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        # BEV 파라미터 (config에서 가져오기)
        self.bev_pixel_size   = config.bev_pixel_size      # meters per pixel
        self.bev_pixel_width  = config.bev_pixel_width     # pixels (가로)
        self.bev_pixel_height = config.bev_pixel_height    # pixels (세로)

        # InternVL3 모델 로드
        # if torch.cuda.is_available() else "cpu"
        self.device = torch.device("cuda")
        print("Using device:", self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, use_fast=False
        )
        self.model = (
            AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            ).eval().to(self.device)

        )
        # Generation config (실시간 점수 산출용)
        self.gen_cfg = dict(max_new_tokens=1024, do_sample=False)

        # 이미지 전처리: 448×448, normalize
        self.preprocess = T.Compose(
            [
                T.Lambda(lambda im: im.convert("RGB")),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    
    def coords_to_pixel(self, local_coords: np.ndarray) -> np.ndarray:
        """
        BEV 로컬 좌표 (x_forward, y_left) [m] → 픽셀 좌표 (u=col, v=row) [px]
        • local_coords[...,0] = x_forward (전방으로 양수)
          local_coords[...,1] = y_left    (왼쪽으로 양수)
        • HD‐Map 이미지에서 원점(0,0)은 하단 중앙: (row = H-1, col = W//2)
        • BEV +X (forward) = 화면 위쪽으로 이동 (row 감소)
          BEV +Y (left)    = 화면 왼쪽으로 이동 (col 감소)
        • BEV_PIXEL_SIZE (m/pixel)로 나눈 뒤 floor() → 정수화
        Returns:
            np.ndarray shape (...,2), dtype=int32: (u, v)
        """
        # 1) 분리
        x_fwd = local_coords[..., 0]  # 전방(+) 축 [m]
        y_lft = local_coords[..., 1]  # 좌측(+) 축 [m]

        # 2) 픽셀 계산
        #    u = (가로 중간 = W/2) − (y_left / pixel_size)
        #    v = (아래 끝 = H−1) − (x_forward / pixel_size)
        u = (self.bev_pixel_width  / 2.0) - (y_lft / self.bev_pixel_size)
        v = (self.bev_pixel_height - 1.0) - (x_fwd / self.bev_pixel_size)

        uv = np.stack([u, v], axis=-1)
        return np.floor(uv).astype(np.int32)

    def _overlay_trajectory(
        self,
        base_img: Image.Image,
        traj_pixels: np.ndarray,
        color: tuple = (255, 0, 0),
        thickness: int = 1,
    ) -> Image.Image:
        """
        단일 trajectory(8점)를 HD-map 위에 오버레이한다.
        Args:
            base_img: PIL Image (RGB), 크기 = (bev_width, bev_height)
            traj_pixels: np.ndarray shape=(8,2), (u,v) 픽셀 좌표
            color: (R,G,B) 튜플
            thickness: 선 굵기 (px)
        Returns:
            PIL Image (RGB) 위에 궤적이 그려진 새로운 이미지
        """
        overlay = base_img.copy()
        draw = ImageDraw.Draw(overlay)
        pts = [(int(u), int(v)) for (u, v) in traj_pixels]
        # PIL draw.line 은 anti-aliasing이 없으므로, cv2로 그려도 됨. 다만 simplicity 를 위해 PIL로 진행.
        draw.line(pts, fill=color, width=thickness)
        return overlay

    def _get_tiles(self, img: Image.Image) -> torch.Tensor:
        img = pad_to_square(img)                    # Letter-box
        img = img.resize((448, 448), Image.BICUBIC) # 왜곡 없이 스케일-업
        tensor = self.preprocess(img)
        return tensor.unsqueeze(0).to(self.device).to(torch.bfloat16)

    def _ask_vlm_for_score(self, hd_img: Image.Image, overlay_img: Image.Image) -> float:
        # 1) 원본 BEV 와 궤적 오버레이 BEV 두 장 준비
        tiles_hd   = self._get_tiles(hd_img)       # (1,3,448,448)
        tiles_traj = self._get_tiles(overlay_img)  # (1,3,448,448)
        # 2) 합치고, 모델에 각각 몇 장인지 알려줌
        pixel_values      = torch.cat([tiles_hd, tiles_traj], dim=0)


        system_prompt = """SYSTEM: You are a vision–language assistant specialized in autonomous driving.

        ASSUMPTIONS:
        • Two images are supplied together:  
        – Image-1 = original HD-map BEV (without trajectory)  
        – Image-2 = same map with a MAGENTA LINE (BGR 255,0,255) showing a candidate trajectory  
        • Pixel resolution = 0.25 m per pixel.

        BEV COLOR LEGEND:
        • Black    = non-drivable background  
        • Light-gray = drivable road  
        • White    = sidewalk / other vehicles (non-drivable)  
        • Yellow   = lane centerline / edge  
        • Orange   = static obstacles (cones, barriers)  
        • Red      = pedestrians  

        DEFINITIONS
        1) **Collision risk**  
        - Step-1 : For every pixel on the magenta path (Image-2) check the *same* (row,col) in Image-1.  
        - Step-2 : If that pixel is **White OR Red OR Orange OR within ±2 pixels (≈0.5 m) of such pixels**, mark as a hit.  
        - If ≥1 hit ⇒ Collision risk = *Yes* (collision_flag = 1); else *No* (collision_flag = 0).  
        - Also report the number of hit points to justify the answer.

        2) **Path smoothness level**  
        • High   (no turn > 30°) → score 1.0  
        • Medium (any turn 30–60°) → score 0.5  
        • Low    (any turn > 60°) → score 0.0  

        3) **Centerline alignment score**
        Output a numeric score ∈ [0.00–1.00] (two decimals) reflecting how closely the MAGENTA path in Image-2 follows the Yellow centerline in Image-1 (higher = better)

        FINAL SCORE = 0.15 × smoothness_score  
                    + 0.4 × centerline_score  
                    + 0.45 × (1 – collision_flag)   // round to 2 decimals.

        TASK → produce **exactly four lines**:

        1) Collision risk: <Yes/No> – <#hit_points> hits on White/Red/Orange within 2 px  
        2) Path smoothness level: <High/Medium/Low> – brief reason  
        3) Centerline alignment score: <0.00–1.00 – reason>
        4) Final score: <X.XX>
        """

        user_prompt = """OUTPUT (exactly four lines):
        Collision risk: <Yes/No – N hits>
        Path smoothness level: <High/Medium/Low – reason>
        Centerline alignment score: <0.00–1.00 – reason>
        Final score: <X.XX>
        """



        full_prompt   = system_prompt + "\n" + user_prompt

        try:
            # 두 장을 batch로 한 번에 넘겨줍니다
            resp = self.model.chat(
                self.tokenizer,
                pixel_values,
                full_prompt,
                self.gen_cfg,
            )
        except Exception:
            traceback.print_exc()
            return 0.0
        resp = resp[0] if isinstance(resp, (list, tuple)) else resp

        # 전체 점수만 parsing
        import re
        m = re.search(r'"Final score"\s*:\s*([0-9]*\.?[0-9]+)', resp)
        if m:
            return float(m.group(1))
        print("[WARN] JSON 파싱 실패:", resp)
        return 0.0

    def select_best_anchor(
        self,
        token: str,
        anchors: np.ndarray,
    ) -> np.ndarray:
        """
        N개의 앵커(각각 shape=(8,2), 단위=m)를 평가하여 가장 점수가 높은 앵커를 반환.
        동시에 다음 파일들을 /data/goodhsm2000/result 에 저장:
         1) 각 앵커별 오버레이 이미지:   {token}_result_anchor_{i}.png
         2) 모든 앵커를 한번에 오버레이한 이미지: {token}_result_all.png
         3) 선택된 베스트 앵커만 오버레이한 이미지: {token}_result_best.png
         4) 앵커별 점수를 기록한 JSONL: {token}_result_scores.jsonl

        Args:
            token: 예) "000123" → "{token}_hd.png" 로 HD‐map을 불러옴
            anchors: np.ndarray (N,8,2) shape, (x_forward, y_left) in meters
        Returns:
            best_anchor: np.ndarray (1,8,2), 형상 그대로 반환
        """
        hd_path = self.hd_map_dir / f"{token}_hd.png"
        if not hd_path.exists():
            raise FileNotFoundError(f"HD‐map not found: {hd_path}")

        # 1) HD-map 불러오기 (PIL RGB)
        hd_img = Image.open(str(hd_path)).convert("RGB")

        # 2) 점수 저장용 JSONL 파일 열기
        scores_path = self.result_dir / f"{token}_result_scores.jsonl"
        fout_scores = open(str(scores_path), "w", encoding="utf-8")

        best_score = -1.0
        best_anchor = None

        # 3) 모든 앵커를 한번에 오버레이하기 위한 복사본
        all_img = hd_img.copy()
        draw_all = ImageDraw.Draw(all_img)

        # 앵커별 색상 팔레트 (8가지 이상의 물감이 필요하면 추가)
        anchor_palette = [
            (255,   0,   0),  # 빨강
            (  0, 255,   0),  # 초록
            (  0,   0, 255),  # 파랑
            (255, 165,   0),  # 주황
            (128,   0, 128),  # 보라
            (  0, 255, 255),  # 시안
            (255, 255,   0),  # 노랑
            (255,   0, 255),  # 마젠타
        ]

        N = anchors.shape[0]
        for i in range(N):
            traj_local = anchors[i]  # shape=(8,2), 단위=m

            # (1) 로컬(m) → 픽셀(px)
            pix = self.coords_to_pixel(traj_local)  # (8,2)

            # (2) 앵커별 이미지 오버레이
            img_i = self._overlay_trajectory(
                base_img = hd_img,
                traj_pixels = pix,
                color = (255, 0, 255),      # 단일 후보는 보라색으로
                thickness=3
            )
            out_i = self.result_dir / f"{token}_result_anchor_{i}.png"
            img_i.save(str(out_i))

            # (3) 동시에 all_img에도 다 그려두기 (다양한 색상 활용)
            color_all = anchor_palette[i % len(anchor_palette)]
            pts_all = [(int(u), int(v)) for (u, v) in pix]
            draw_all.line(pts_all, fill=color_all, width=1)

            # (4) VLM에 질의하여 score 얻기
            print(f"[{token}] Evaluating anchor {i+1}/{N} ...", end="", flush=True)
            score_i = self._ask_vlm_for_score(hd_img, img_i)
            print(f"  score={score_i:.2f}")

            # (5) JSONL에 기록
            record = {"anchor_idx": i, "score": score_i}
            fout_scores.write(json.dumps(record, ensure_ascii=False) + "\n")

            # (6) 베스트 앵커 갱신
            if score_i > best_score:
                best_score = score_i
                best_anchor = traj_local.copy()

        fout_scores.close()

        all_out = self.result_dir / f"{token}_result_all.png"
        all_img.save(str(all_out))

        # 5) 베스트 앵커만 오버레이된 이미지 저장
        if best_anchor is not None:
            pix_best = self.coords_to_pixel(best_anchor)  # (8,2)
            img_best = self._overlay_trajectory(
                base_img = hd_img,
                traj_pixels = pix_best,
                color = (255, 0, 0),
                thickness = 2
            )
            best_out = self.result_dir / f"{token}_result_best.png"
            img_best.save(str(best_out))
        else:
            # 혹시 best_anchor가 None인 경우에도 기본적으로 첫 번째 앵커를 복사
            best_anchor = anchors[0:1]
            pix_best = self.coords_to_pixel(best_anchor[0])
            img_best = self._overlay_trajectory(
                base_img = hd_img,
                traj_pixels = pix_best,
                color = (255, 0, 0),
                thickness=2
            )

            best_out = self.result_dir / f"{token}_result_best.png"
            img_best.save(str(best_out))

        return best_anchor.reshape(1, 8, 2)  # shape = (1,8,2)



# ================ 테스트용 예시 스크립트 ================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HD‐Map 위에 각 앵커 궤적을 오버레이하고, InternVL3으로 스코어링"
    )
    parser.add_argument(
        "--token",
        type=str,
        default= "0df6f24a95e75544",
        help="토큰 ID (예: '000123') → '{token}_hd.png' 로드"
    )
    parser.add_argument(
        "--anchor_npy",
        type=str,
        default="/data/goodhsm2000/repos/DiffusionDrive/kmeans_navsim_traj_20.npy",
        help="(N,8,2) 형태의 앵커 궤적(.npy) 파일 경로"
    )

    args = parser.parse_args()


    # 2) 앵커 불러오기
    anchors_np = np.load(args.anchor_npy)  # shape = (N,8,2)

    # 3) Scorer 생성 및 실행
    scorer = AnchorTrajectoryScorer(
        config=TransfuserConfig,
        model_name = "OpenGVLab/InternVL2_5-4B",
        hd_map_dir = "/data/goodhsm2000/repos/InternVL/HD_Map",
        result_dir = "/data/goodhsm2000/result",
    )
    best_anchor = scorer.select_best_anchor(
        token   = args.token,
        anchors = anchors_np,
    )
    print(f"▶ Best anchor (1×8×2):\n{best_anchor}")
