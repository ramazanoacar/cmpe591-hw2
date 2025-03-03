# HW2

## Training Process

For our training implementation, we used the following parameters:

```python
num_episodes = 3800
update_frequency = 10
target_update_frequency = 200
epsilon = 1.0
epsilon_decay = 0.997
epsilon_min = 0.03
batch_size = 64
gamma = 0.99
buffer_size = 100000
learning_rate = 1e-3
```

We conducted training with a maximum timestep of 100. After an extended training session using the instructor's parameters, we realized we hadn't saved the data properly, resulting in empty outputs. Learning from this experience, we updated the epsilon decay and minimum epsilon values as specified above.

![DQN Training Plot](/dqn_training_plot.png)

## Performance Analysis

### Overall Training Performance
The graph above illustrates our agent's reward performance throughout the training process. We observed an upward trend in average rewards as training progressed, indicating successful learning.

### Reward Progression
![Reward](/reward.png)
This graph displays the total rewards obtained by our agent for each episode. The increasing trend demonstrates the agent's improving ability to maximize rewards over time.

### Reward Per Step
![RPS](/rps.png)
This visualization shows the average rewards received per step. As the learning process stabilized, we observed a consistent upward trend in per-step rewards, suggesting more efficient decision-making.

### Smoothed Reward (Window Size: 50)
![Smoothed Reward 50](/smoothed_reward_50.png)
Using a moving average with a window size of 50, this graph reveals a more consistent pattern in reward progression. The smoothing helps filter out noise and highlights the steady improvement throughout training.

### Smoothed Reward (Window Size: 100)
![Smoothed Reward 100](/smoothed_reward_100.png)
With a larger window size of 100, the reward trend becomes even clearer. The graph demonstrates a definitive upward trajectory in average rewards as training advances, confirming effective learning.

### Smoothed Reward Per Step (Window Size: 50)
![Smoothed RPS 50](/smoothed_rps_50.png)
This graph shows the moving average of rewards per step with a window size of 50. The steady increase indicates that our agent became progressively more efficient at maximizing rewards with each action.

### Smoothed Reward Per Step (Window Size: 100)
![Smoothed RPS 100](/smoothed_rps_100.png)
With a larger window size, the general trend in per-step rewards becomes more apparent. The consistent improvement over time demonstrates that our agent developed increasingly optimal policies for the environment.
