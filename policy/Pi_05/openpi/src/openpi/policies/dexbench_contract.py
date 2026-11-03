"""Shared training/checkpoint/inference contract for the prepared DexBench data."""
ACTION_CONTRACT = {
    'schema': 'dexbench-pi05-absolute-ee-wuji20-v1',
    'action_type': 'ee', 'action_mode': 'absolute', 'state_mode': 'absolute',
    'pose': 'world_xyz_wxyz', 'hand': 'absolute_wuji_hand2_joints',
    'hand_order': 'isaac_stage_major_index_middle_pinky_ring_thumb',
    'layout': [['left_ee_pose', 7], ['left_ee_joint_state', 20],
               ['right_ee_pose', 7], ['right_ee_joint_state', 20]],
    'cameras': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
    'color': 'RGB', 'prompt': '[{embodiment}/{task}] {instruction}',
}
