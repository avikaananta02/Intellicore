for step in range(1, config.TOTAL_STEPS + 1):

    action = agent.select_action(state)

    next_obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    next_state = stacker.append(
        preprocessor.process(next_obs)
    )

    agent.store_transition(
        state,
        action,
        reward,
        next_state,
        done,
    )

    loss = agent.learn()

    if loss is not None:
        losses.append(loss)

    episode_reward += reward

    state = next_state

    if step % config.TARGET_UPDATE == 0:
        agent.sync_target()

    if step % config.SAVE_EVERY == 0:
        agent.save(
            f"checkpoint_{step}.pth"
        )

    if done:

        episode_rewards.append(
            episode_reward
        )

        episode_count += 1

        print(
            f"Episode {episode_count} | "
            f"Reward: {episode_reward}"
        )

        obs, info = env.reset()

        state = stacker.reset(
            preprocessor.process(obs)
        )

        episode_reward = 0.0
