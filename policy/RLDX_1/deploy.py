"""XPolicyLab environment loops for RLDX_1.

The RPC sequence intentionally stays identical to ``policy/demo_policy`` so
the websocket transport and debug environment exercise the standard path.
"""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()
        model_client.call(func_name="update_obs", obs=obs)
        actions = model_client.call(func_name="get_action")

        for action_idx, action in enumerate(actions):
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end() or action_idx + 1 == len(actions):
                break
            # Retain every executed frame.  The adapter samples the official
            # four-frame, stride-two RLDX video history on its next inference.
            model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")

    while not TASK_ENV.is_episode_end():
        env_idx_list = [int(i) for i in TASK_ENV.get_running_env_idx_list()]
        if not env_idx_list:
            break
        obs_list = TASK_ENV.get_obs_batch(env_idx_list)
        model_client.call(func_name="update_obs_batch", obs=obs_list)
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)

        if not actions or len(actions) != len(env_idx_list):
            raise ValueError("RLDX batch result count does not match active environments")
        chunk_size = len(actions[0])
        if chunk_size <= 0 or any(len(chunk) != chunk_size for chunk in actions):
            raise ValueError("RLDX batch action chunks must be non-empty and equally sized")
        for action_idx in range(chunk_size):
            current_action_list = [env_actions[action_idx] for env_actions in actions]
            TASK_ENV.take_action_batch(current_action_list, env_idx_list)

            if TASK_ENV.is_episode_end() or action_idx + 1 == chunk_size:
                break

            running = set(TASK_ENV.get_running_env_idx_list())
            active_batch_idx = [
                i for i, env_idx in enumerate(env_idx_list) if env_idx in running
            ]
            actions = [actions[i] for i in active_batch_idx]
            env_idx_list = [env_idx_list[i] for i in active_batch_idx]
            if not env_idx_list:
                break
            model_client.call(
                func_name="update_obs_batch",
                obs=TASK_ENV.get_obs_batch(env_idx_list),
            )
