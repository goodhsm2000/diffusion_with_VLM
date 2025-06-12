## 📁 Final Code Locations

The final implementation can be found under the following directory:
**diffusion_with_VLM/navsim/agents/diffusiondrive/**


| Method Combination                                                                 | File Name                                                                                                                      |
|------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| **Dynamic Anchor Sampling (VLM)** + DiffusionDrive + **Trajectory Scorer (VLM)**  | [`transfuser_model_v2.py`](diffusion_with_VLM/navsim/agents/diffusiondrive/transfuser_model_v2.py)                           |
| **Dynamic Anchor Sampling (YOLO)** + DiffusionDrive + **Trajectory Scorer (VLM)** | [`transfuser_model_v2_yolo_vlmscorer.py`](diffusion_with_VLM/navsim/agents/diffusiondrive/transfuser_model_v2_yolo_vlmscorer.py) |
| **Dynamic Anchor Sampling (VLM)** + DiffusionDrive                                | [`transfuser_model_v2_vlm_DA.py`](diffusion_with_VLM/navsim/agents/diffusiondrive/transfuser_model_v2_vlm_DA.py)             |
| **Dynamic Anchor Sampling (YOLO)** + DiffusionDrive                               | [`transfuser_model_v2_yolo_DA.py`](diffusion_with_VLM/navsim/agents/diffusiondrive/transfuser_model_v2_yolo_DA.py)                 |
| DiffusionDrive only                                                               | [`transfuser_model_v2_copy.py`](diffusion_with_VLM/navsim/agents/diffusiondrive/transfuser_model_v2_copy.py)                 |
