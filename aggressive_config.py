SHARED_CORE_ENV_ID = "highway-v0"

AGGRESSIVE_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
        "clip": True,
        "see_behind": True,
        "observe_intentions": False,
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": [20, 25, 30],
    },
    "lanes_count": 4,
    "vehicles_count": 45,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "duration": 30,
    "ego_spacing": 2,

    "vehicles_density": 0.6,       
    "collision_reward": -1.5,        
    "right_lane_reward": 0.1,        
    "high_speed_reward": 1.2,         
    "lane_change_reward": -0.1,      
    "reward_speed_range": [25, 30],   
    
    "normalize_reward": True,
    "offroad_terminal": True,
}
