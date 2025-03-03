# HW2


## Training

You can train your DQN model using the following command:
```python
from hw2_yy import train, test
import torch

# Train
policy_net, rewards, rps = train(num_episodes=1000)
```


## Testing

You can test the model using the following command:
```python
from hw2_yy import train, test
import torch

# Load
avg = test(num_episodes=5)
```

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

We conducted training with a maximum timestep of 100. After an extended training session using the instructor's parameters, we realized we hadn't saved the data properly, resulting in empty outputs. Learning from this experience, we updated the epsilon decay and minimum epsilon values as specified above. Since the data is fluctuating a lot we also smoothed the data for better understanding the pattern. 

![DQN Training Plot](/dqn_training_plot.png)

## Performance Analysis

### Overall Training Performance
The graph above illustrates our agent's reward performance throughout the training process. We observed an upward trend in average rewards as training progressed, indicating successful learning.

### Reward Progression
![Reward](/reward.png)
This graph displays the total rewards obtained by our agent for each episode. Its hard to comment on this data since it fluctuates a lot. We made smoothed version of this data for better visualisation.

### Reward Per Step
![RPS](/rps.png)
This visualization shows the average rewards received per step. As the learning process stabilized, we observed an upward trend in per-step reward. Also since data fluctuates a lot its hard to read so we put this image as well as the smoothed version. 

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
