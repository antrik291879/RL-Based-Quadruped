Multi-Terrain Adaptation of quadruped through Reinforcement Learning
===

<img width="896" height="479" alt="image" src="https://github.com/user-attachments/assets/144392b5-ba5e-42cb-bcf4-912cbefa8f46" />

In Today's era quadruped. are no londer just pre‑programmed gaits or fragile dynamic models.The are Learning how to navigate the real world with an agility once reserved for animals.Breakthrough in deep reinforcement learning, massive parallel simulation.

Tasks that were once considered impossible for robots such as climbing on stairs are now demonstrated by policies trained entirely in simulation.We are seeing this coming to real.

We prefer Reinforcement Learning for quadruped terrain adaptation because the core problem is not one of calculation; it's one of adaptation and robustness in the face of near-infinite, un-modellable uncertainty.

Classical control fails because its foundational premise that you can accurately model the system and its environment is fundamentally untrue for unstructured terrain.

Overview
===
This project aims to develop a reinforcement learning (RL)-based locomotion controller for a quadruped robot (Unitree Go1) capable of traversing diverse terrains (flat, gravel, grass, slopes, stairs, and soft surfaces).

* Walk stably on flat ground.
* Ascend and descend on stairs.
* Recover from unexpected slips and pushes.

The Robot entirely in simulation using Proximal Policy Optimization (PPO) on NVIDIA Isaac Gym, within our custom high‑throughput training framework HIMLoco.We analyse the training by :-

* Mean episodic reward and its convergence rate over thousands of iterations.
* Episode survival rate – fraction of episodes without early termination (falls, excessive tilt).

1.Reinforcement Learning
===
Reinforcement Learning (RL) is a computational framework where an agent learns to make sequential decisions by interacting with an environment.

Every locomotion problem can be framed as a Markov Decision Process, defined by the tuple (S,A,P,R,γ)

The agent’s goal is to learn a policy πθ(a∣s) which maps the state to action.

* deterministic policy a = π(s)
* Stochastic policy π(a|s)=P[At=a|St=s]

Value Function Vπ(s) = Eπ[Rt+1+γRt+2+....|St=s]
* Prediction of future rewards
* How good/bad a state is

2.Proximal Policy Optimization (PPO)
===

PPO is an actor‑critic policy gradient algorithm that addresses the key challenge of policy gradient methods: they tend to be unstable if the policy changes too much in a single update. PPO ensures stable learning by “clipping” the update, preventing the new policy from deviating far from the old one.

The value function is used to compute the advantage. We use Generalized Advantage Estimation (GAE), which balances bias and variance with parameter λ.

The value function is optimised by minimising the mean squared error between its predictions and the actual returns (computed via GAE).

3.Learning to Walk
==
Teaching a quadruped to walk with reinforcement learning boils down to defining four essential ingredients. Each shapes the robot’s behaviour and together they determine whether the final policy stumbles or strides confidently into the real world.

3.1 Actions
==
The policy directly controls the robot by outputting desired joint positions for all 12 motors — three per leg: hip, thigh, and calf. Each value is a normalised number between –1 and 1, remapped to a physical joint angle before being sent to a low‑level PD controller that converts the position error into motor torque. This position‑based interface acts as a natural shield against unmodelled hardware quirks and is the standard for reliable sim‑to‑real transfer.

3.2 Observations
==
The observation vector tells the robot everything it can feel about itself and the ground without using cameras. We feed it a 48‑dimensional stream.All signals are exactly what the physical Go1 can measure onboard.

3.3 Reward Design
==
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_joint_power(self):
        #Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

Results
===
<img width="1331" height="679" alt="image" src="https://github.com/user-attachments/assets/558ea59b-a45c-40aa-b2d3-1c0a47ade25e" />



<img width="888" height="476" alt="image" src="https://github.com/user-attachments/assets/1042feb4-c370-482c-903f-49e41ffee8a6" />

<img width="902" height="507" alt="image" src="https://github.com/user-attachments/assets/9cdafdb7-ad71-47a8-ac9e-aaeaff53f0ca" />





