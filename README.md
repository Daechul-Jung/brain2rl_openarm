# brain2rl_openarm
brain2rl_openarm wrapper
```
$HOME/
└─ ros2_ws/                      # ROS 2 workspace (colcon build)
   ├─ src/
   │  ├─ openarm_ros2/           # upstream repo (bringup, moveit config, hardware*)
   │  │  ├─ openarm_bringup/     # launch/config (we use for sim)
   │  │  ├─ openarm_bimanual_moveit_config/  # MoveIt config (fake controller demo)
   │  │  └─ openarm_hardware/  
   │  ├─ openarm_description/    # URDF/Xacro + meshes for OpenArm
   │  └─ brain2rl_openarm/       # ★ your ROS wrapper for RL + sim glue
   │     ├─ package.xml
   │     ├─ setup.py
   │     ├─ resource/
   │     │  └─ brain2rl_openarm  # ament marker file (must exist)
   │     ├─ brain2rl_openarm/
   │     │  ├─ __init__.py
   │     │  ├─ envs/
   │     │  │  └─ openarm_env.py           # Gym-style env 
   │     │  ├─ scripts/
   │     │  │  ├─ rl_train_joint.py        # minimal trainer using joint deltas 
   │     │  │  ├─ run_with_agent.py        # call my rl algorithm agents
   │     │  │  └─ spawn_random_cup.py      # Gazebo: spawn a cylinder “cup”
   │     │  ├─ utils/                     
   │     │  ├─ config/
   │     │  │  └─ openarm_controllers.yaml 
   │     │  └─ launch/
   │     │     ├─ openarm_rl_gazebo.launch.py  # starts Gazebo + controllers + cup + trainer
   │     │     └─ openarm_demo.launch.py      
   │     └─ test/                       
   └─ (build/ install/ log/ after you build)
```