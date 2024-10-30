"""
A simple implementation of the entire pipeline
to help understand the idea
"""
import random
import numpy as np
import jax.numpy as jnp
import jax
import equinox as eqx
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt

# N=10 random actions
actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N = 10

def sample_actions(curr):
    """
    curr is an int representing the current position.
    for any curr, there are two possible actions.
    curr-1 and curr*2 mod N
    except for curr=1
    """
    return jnp.where(curr == 1, jnp.array([N, curr * 2 % (N+1)]), jnp.array([curr - 1, curr * 2 % (N+1)]))

def random_action(actions):
    """
    Randomly select an action from the list of actions.
    """
    return random.choice(actions)

def loss_fn_policy(policy_net, advantage_net, states, final_reward, goal):
    baseline = jax.vmap(advantage_net)(states)
    advantage = final_reward - baseline

    states = jnp.concatenate([states, goal], axis=1)
    log_probs = jax.vmap(policy_net)(states)
    policy_loss = -jnp.mean(log_probs * advantage)

    return policy_loss

def loss_fn_advantage(advantage_net, states, final_reward):
    baseline = jax.vmap(advantage_net)(states)
    advantage_loss = jnp.mean((baseline - final_reward) ** 2)

    return advantage_loss


class PolicyNW(eqx.Module):
    layers: list
    def __init__(self, input=2, output=2):
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(input, 10, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(10, output, key=key2),
            jax.nn.softmax
            ]

    @eqx.filter_jit
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class AdvantageNW(eqx.Module):
    layers: list
    def __init__(self, input=2, output=1):
        key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(input, 10, key=key1, use_bias=False),
            jax.nn.relu,
            eqx.nn.Linear(10, output, key=key2, use_bias=False)
            ]

    @eqx.filter_jit
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

N_ITERATIONS = 100


policy = PolicyNW(3, 2)
advantage = AdvantageNW(2, 2)



iteration_cache = []
success_cache = []


N_WORKERS = 64

for epoch in range(100):
    total_iterations = 0
    success_count = 0

    observations = jnp.array([(np.random.randint(N)) for _ in range(N_ITERATIONS)], dtype=jnp.int8)
    goals = jnp.array([(np.random.randint(N)) for obs in observations], dtype=jnp.int8)

    valid_episodes = [(obs, goal) for obs, goal in zip(observations, goals) if obs != goal]

    def process_episode(observation, goal):
        predictions = []
        goal = jnp.array([goal], dtype=jnp.int8)
        
        iteration = 0
        total_reward = 0
        success = False
        while iteration < 100:
            curr = observation

            valid_actions = jnp.array(sample_actions(curr))
            obs = jnp.concatenate([valid_actions, goal])
            prediction = policy(obs)
            predictions.append(prediction)
            action = valid_actions[jnp.argmax(prediction)]

            observation = action
            iteration += 1

            reward = -0.01 * iteration
            if curr == goal[0]:
                success = True
                reward = 25.0
                total_reward += reward
                return predictions, total_reward, iteration, success, goal
            
            total_reward += reward

        return predictions, total_reward, iteration, success, goal
    
    count = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(process_episode, obs, goal) for obs, goal in valid_episodes]
        
        for future in as_completed(futures):
            count += 1
            
            # print("Completed {}".format(count))
            predictions, total_reward, iterations, success, goal = future.result()
            predictions = jnp.array(predictions)
            goal_arr = jnp.repeat(goal, predictions.shape[0], axis=0).reshape(predictions.shape[0], 1)
            total_reward = jnp.float32(total_reward)
            
            policy_loss_value, policy_grads = eqx.filter_value_and_grad(loss_fn_policy)(policy, advantage, predictions, total_reward, goal_arr)
            advantage_loss_value, advantage_grads = eqx.filter_value_and_grad(loss_fn_advantage)(advantage, predictions, total_reward)

            total_iterations += iterations
            success_count += int(success)
        
    iteration_cache.append(total_iterations / success_count if success_count > 0 else 100)
    success_cache.append(success_count / N_ITERATIONS * 100)

    
    print(f"Epoch: {epoch+1} Completed")

print(iteration_cache)
print(success_cache)

# Save average iterations to reach goal vs epoch(the index represents the epoch):
# let x be epoch and y be average iterations to reach goal
plt.figure()
plt.plot(range(len(iteration_cache)), iteration_cache)
plt.xlabel("Epoch")
plt.ylabel("Average Iterations to reach goal")
plt.title("Average Iterations to Reach Goal vs Epoch")
plt.grid(True)
plt.savefig("/home/aditkadepurkar/dev/MCTS-Env/average_iterations_vs_epoch.png")

plt.figure()
plt.plot(range(len(success_cache)), success_cache)
plt.xlabel("Epoch")
plt.ylabel("Success Rate (%)")
plt.title("Success Rate vs Epoch")
plt.grid(True)
plt.savefig("/home/aditkadepurkar/dev/MCTS-Env/success_rate_vs_epoch.png")
