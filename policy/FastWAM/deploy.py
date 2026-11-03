def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        obs = TASK_ENV.get_obs() # Get Observation
        model_client.call(func_name="update_obs", obs=obs)  # Update Observation, `update_obs` here can be modified
        
        actions = model_client.call(func_name="get_action") # Get Action according to observation chunk
        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break
            
            # A simulator may finish an episode in the middle of a predicted
            # chunk.  Do not request a post-terminal observation; some real
            # backends invalidate their observation handles immediately.
            if action_idx != len(actions) - 1:
                obs = TASK_ENV.get_obs() # Get Observation
                model_client.call(func_name="update_obs", obs=obs)

def eval_one_episode_batch(TASK_ENV, model_client):

    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end(): # Check whether the episode ends
        env_idx_list = [int(env_idx) for env_idx in TASK_ENV.get_running_env_idx_list()]
        if not env_idx_list:
            break
        obs_list = TASK_ENV.get_obs_batch(env_idx_list) # Get Batch Observation

        model_client.call(func_name="update_obs_batch", obs=obs_list)  # Update Observation, `update_obs` here can be modified
        # XPolicyLab transports the batch indices through the legacy-compatible
        # `obs` field; the model method receives them as env_idx_list.
        actions = model_client.call(func_name="get_action_batch", obs=list(env_idx_list))

        if not actions:
            raise ValueError("FastWAM get_action_batch returned no action chunks")
        if len(actions) != len(env_idx_list):
            raise ValueError(
                "FastWAM batch result count does not match env_idx_list: "
                f"{len(actions)} != {len(env_idx_list)}"
            )
        chunk_size = len(actions[0])
        if chunk_size <= 0 or any(len(chunk) != chunk_size for chunk in actions):
            raise ValueError("FastWAM batch action chunks must be non-empty and equally sized")
        for action_idx in range(chunk_size):
            current_action_list = [env_actions[action_idx] for env_actions in actions]

            TASK_ENV.take_action_batch(current_action_list, env_idx_list)
            
            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_idx = [i for i, env_idx in enumerate(env_idx_list) if env_idx in running]

            actions = [actions[i] for i in active_batch_idx]
            env_idx_list = [env_idx_list[i] for i in active_batch_idx]
            if not env_idx_list:
                break
            model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(env_idx_list))
