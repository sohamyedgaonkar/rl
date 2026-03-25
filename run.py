
import os
os.environ['BEARTYPE_CLAW'] = 'false'

import random
import json
from collections import defaultdict

from server.my_env_environment import WordGameEnvironment
from server.models import WordGameAction, WordGameObservation, WordGameState

WORDS = "qwertyuiopasdfghjklzxcvbnm"

class RandomPolicy:
    """Pure random — baseline."""
    def select_action(self, observation: WordGameObservation) -> str:
        available = [c for c in WORDS if c not in observation.guessed_letters]
        return random.choice(available) if available else random.choice(WORDS)

class QLearningPolicy:
    """Q-learning policy for word guessing."""
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = defaultdict(lambda: defaultdict(float))  # Q(state, action)
        self.last_state = None
        self.last_action = None
        
    def get_state_key(self, observation: WordGameObservation) -> str:
        """Create a state key from the observation."""
        return observation.masked_word
    
    def select_action(self, observation: WordGameObservation) -> str:
        state = self.get_state_key(observation)
        available_actions = [c for c in WORDS if c not in observation.guessed_letters]
        
        if not available_actions:
            return random.choice(WORDS)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action = random.choice(available_actions)
        else:
            # Choose action with highest Q-value
            q_values = [(a, self.q_table[state][a]) for a in available_actions]
            action = max(q_values, key=lambda x: x[1])[0]
        
        self.last_state = state
        self.last_action = action
        return action
    
    def update(self, observation: WordGameObservation, reward: float, done: bool):
        """Update Q-values based on the transition."""
        if self.last_state is None or self.last_action is None:
            return
            
        state = self.last_state
        action = self.last_action
        next_state = self.get_state_key(observation)
        
        # Q-learning update
        current_q = self.q_table[state][action]
        
        if done:
            next_max_q = 0
        else:
            available_next = [c for c in WORDS if c not in observation.guessed_letters]
            if available_next:
                next_max_q = max(self.q_table[next_state][a] for a in available_next)
            else:
                next_max_q = 0
        
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q
        
        self.last_state = None
        self.last_action = None
    
    def save_q_table(self, filename="q_table.json"):
        """Save Q-table to file."""
        # Convert defaultdict to regular dict for JSON serialization
        q_dict = {state: dict(actions) for state, actions in self.q_table.items()}
        with open(filename, 'w') as f:
            json.dump(q_dict, f, indent=2)
    
    def load_q_table(self, filename="q_table.json"):
        """Load Q-table from file."""
        try:
            with open(filename, 'r') as f:
                q_dict = json.load(f)
                self.q_table = defaultdict(lambda: defaultdict(float))
                for state, actions in q_dict.items():
                    for action, value in actions.items():
                        self.q_table[state][action] = value
        except FileNotFoundError:
            pass

class FrequencyPolicy:
    """Policy based on letter frequency in English words."""
    
    # Letter frequencies in English (most common first)
    LETTER_FREQUENCY = "etaoinsrhdlucmfywgpbvkxqjz"
    
    def select_action(self, observation: WordGameObservation) -> str:
        available = [c for c in self.LETTER_FREQUENCY if c not in observation.guessed_letters]
        return available[0] if available else random.choice([c for c in WORDS if c not in observation.guessed_letters] or WORDS)

def run_episode(env, policy, train=True, verbose=False):
    """Play one episode. Returns 1 if caught, 0 if missed."""
    result = env.reset()
    step = 0
    total_reward = 0

    while not result.done:
        action_id = policy.select_action(result)
        next_result = env.step(WordGameAction(guess=action_id))
        step += 1
        
        if train and hasattr(policy, 'update'):
            policy.update(next_result, next_result.reward or 0, next_result.done)
        
        total_reward += next_result.reward or 0
        
        if verbose:
            print(f'  Step {step}: Guessed "{action_id}" - {next_result.message}')
        
        result = next_result

    caught = 1 if result.reward and result.reward > 0 else 0
    if verbose:
        status = 'Caught!' if caught else 'Missed'
        print(f'  Result: {status} (reward={result.reward})')
    return caught, total_reward


def train_policy(policy_class, episodes=1000, save_path=None):
    """Train a policy and return it."""
    env = WordGameEnvironment()
    policy = policy_class()
    
    if hasattr(policy, 'load_q_table') and save_path:
        policy.load_q_table(save_path)
    
    total_caught = 0
    total_reward = 0
    
    for ep in range(episodes):
        if ep % 100 == 0:
            print(f'Training episode {ep+1}/{episodes}')
        
        caught, reward = run_episode(env, policy, train=True, verbose=False)
        total_caught += caught
        total_reward += reward
    
    if hasattr(policy, 'save_q_table') and save_path:
        policy.save_q_table(save_path)
    
    print(f'Training complete: {total_caught}/{episodes} caught ({total_caught/episodes:.2%})')
    print(f'Average reward: {total_reward/episodes:.3f}')
    
    return policy

def evaluate_policy(policy, episodes=100):
    """Evaluate a trained policy."""
    env = WordGameEnvironment()
    total_caught = 0
    total_reward = 0
    
    for ep in range(episodes):
        if ep % 10 == 0:
            print(f'Evaluation episode {ep+1}/{episodes}')
        
        caught, reward = run_episode(env, policy, train=False, verbose=False)
        total_caught += caught
        total_reward += reward
    
    print(f'Evaluation: {total_caught}/{episodes} caught ({total_caught/episodes:.2%})')
    print(f'Average reward: {total_reward/episodes:.3f}')
    return total_caught / episodes

def demonstrate_policies():
    """Demonstrate different policies on the same word."""
    # Load the trained Q-learning policy
    q_policy = QLearningPolicy()
    q_policy.load_q_table("q_table.json")
    
    # Get a consistent word for demonstration
    env = WordGameEnvironment()
    result = env.reset(seed=42)
    target_word = env._target
    
    print(f"\nDemonstrating policies on word: {target_word}")
    print(f"Masked word: {result.masked_word}")
    
    policies = [
        ("Random", RandomPolicy()),
        ("Frequency", FrequencyPolicy()),
        ("Q-Learning", q_policy),
    ]
    
    for name, policy in policies:
        print(f"\n--- {name} Policy ---")
        env_demo = WordGameEnvironment()
        result_demo = env_demo.reset(seed=42)  # Same word for fair comparison
        steps = 0
        
        while not result_demo.done and steps < 20:  # Limit steps for demo
            action = policy.select_action(result_demo)
            result_demo = env_demo.step(WordGameAction(guess=action))
            steps += 1
            print(f"Step {steps}: Guess '{action}' -> {result_demo.message}")
            if result_demo.done:
                break
        
        if result_demo.done and result_demo.reward == 1.0:
            print(f"Success in {steps} steps!")
        else:
            print(f"Failed or ran out of attempts")

if __name__ == "__main__":
    # print("Training Q-Learning Policy...")
    # q_policy = train_policy(QLearningPolicy, episodes=2000, save_path="q_table.json")
    
    # print("\nEvaluating Q-Learning Policy...")
    # evaluate_policy(q_policy, episodes=100)
    
    # print("\nEvaluating Frequency Policy...")
    # freq_policy = FrequencyPolicy()
    # evaluate_policy(freq_policy, episodes=100)
    
    # print("\nEvaluating Random Policy...")
    # random_policy = RandomPolicy()
    # evaluate_policy(random_policy, episodes=100)
    
    # Demonstrate on a specific word
    demonstrate_policies()
