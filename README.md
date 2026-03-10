# RL-Based-Quadruped

This project aims to develop a reinforcement learning (RL)-based locomotion 
controller for a quadruped robot (Unitree Go1) capable of traversing diverse 
terrains (flat, gravel, grass, slopes, stairs, and soft surfaces) without 
terrain-specific retraining. 
The system uses a hierarchical RL framework: 
● High-level policy selects gait and velocity based on proprioception and 
terrain history. 
● Low-level policy outputs joint torques for robust, adaptive stepping. 
The policy is trained in simulation (Isaac Gym) and deployed zero-shot or with 
minimal fine-tuning on real hardware. The project includes hardware validation, 
ablation studies, and novel contributions (e.g., terrain-agnostic reward shaping, 
fall recovery, energy efficiency).
