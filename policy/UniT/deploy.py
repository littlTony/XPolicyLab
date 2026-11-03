"""Standard XPolicyLab episode loop, respecting the configured execution length."""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    execute_steps = int(model_client.call(func_name="get_metadata")["execute_steps"])
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")
        for action in actions[:execute_steps]:
            TASK_ENV.take_action(action)
            if TASK_ENV.is_episode_end():
                break
            model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    execute_steps = int(model_client.call(func_name="get_metadata")["execute_steps"])
    while not TASK_ENV.is_episode_end():
        indices = TASK_ENV.get_running_env_idx_list()
        model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(indices))
        actions = model_client.call(func_name="get_action_batch", obs=indices)
        for step in range(min(len(actions[0]), execute_steps)):
            TASK_ENV.take_action_batch([chunk[step] for chunk in actions], indices)
            if TASK_ENV.is_episode_end():
                break
            running = set(TASK_ENV.get_running_env_idx_list())
            active = [index for index, env in enumerate(indices) if env in running]
            actions = [actions[index] for index in active]
            indices = [indices[index] for index in active]
            model_client.call(func_name="update_obs_batch", obs=TASK_ENV.get_obs_batch(indices))
