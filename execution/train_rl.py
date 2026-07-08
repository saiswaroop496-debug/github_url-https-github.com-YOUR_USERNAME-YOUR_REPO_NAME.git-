import torch
import numpy as np
import time
from pathlib import Path
from exchange_simulator import BettingExchangeEnv
from rl_agent import SoftActorCriticAgent

def train_sac_agent(num_episodes=500, max_steps_per_episode=100, batch_size=256, save_dir="model_versions/latest"):
    """
    Offline Reinforcement Learning Training Loop for the V9 Omni-Quant SAC Agent.
    Allows the agent to interact with the simulated Betting Exchange to learn optimal
    execution strategies (tranche sizing and urgency) based on historical/simulated tick data.
    """
    print("=" * 55)
    print("  [V9 MLOps] Initiating SAC Agent RL Training")
    print("=" * 55)
    
    # Initialize Environment and Agent
    env = BettingExchangeEnv(initial_capital=10000.0, max_steps=max_steps_per_episode)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SoftActorCriticAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=256)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    returns = []
    
    start_time = time.time()
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Execute action in the Limit Order Book Simulator
            next_state, reward, done, info = env.step(action)
            
            # Store transition in Replay Buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            
            # Update network weights (Actor, Critic, and Entropy Temperature)
            if len(agent.replay_buffer) > batch_size:
                agent.update_parameters(batch_size=batch_size)
                
        returns.append(episode_reward)
        
        if (episode + 1) % 50 == 0:
            avg_return = np.mean(returns[-50:])
            elapsed = time.time() - start_time
            print(f"  Episode {episode+1}/{num_episodes} | Avg Reward (Last 50): {avg_return:.2f} | Capital: ${info['capital']:.2f} | Time: {elapsed:.1f}s")
            
    # Save trained weights
    print(f"\n  [OK] Training Complete. Saving SAC weights to {save_dir}...")
    torch.save(agent.actor.state_dict(), Path(save_dir) / "sac_actor.pth")
    torch.save(agent.critic.state_dict(), Path(save_dir) / "sac_critic.pth")
    
    print("  [OK] V9 SAC Agent successfully deployed for production inference.")

if __name__ == "__main__":
    train_sac_agent(num_episodes=500, max_steps_per_episode=100)
