"""
RecurrentPPO Test Demo Runner
Tests the LSTM model across all difficulty levels and produces a detailed diagnostic report.
"""
import os
import sys
import numpy as np
import time

# Must import these BEFORE loading the model so SB3 can find the custom extractor
from hivemind_env.env import HiveMindSingleAgentEnv
from hivemind_env.models import CustomCombinedExtractor

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    print("ERROR: sb3_contrib not installed. Run: pip install sb3-contrib")
    sys.exit(1)

def run_evaluation(model_path, difficulty, num_episodes=10, render=False):
    """Run the model for N episodes at a given difficulty and collect metrics."""
    print(f"\n{'='*60}")
    print(f"  Testing at Difficulty Level {difficulty} ({num_episodes} episodes)")
    print(f"{'='*60}")
    
    render_mode = "human" if render else None
    env = HiveMindSingleAgentEnv(render_mode=render_mode, difficulty_level=difficulty)
    
    # Load the RecurrentPPO model
    model = RecurrentPPO.load(model_path, device="cpu")
    
    results = []
    action_counts = np.zeros(7, dtype=int)
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        picked_up = False
        dropped_off = False
        collided = False
        ep_actions = []
        
        # RecurrentPPO requires LSTM states to be tracked
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        
        while not done:
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states, 
                episode_start=episode_start,
                deterministic=True
            )
            episode_start = np.zeros((1,), dtype=bool)  # Only true at first step
            
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1
            ep_actions.append(int(action))
            action_counts[int(action)] += 1
            
            # Track what happened
            if reward >= 1.5:  # Pickup gives +2.0
                picked_up = True
            if reward >= 9.0:  # Dropoff gives +10.0
                dropped_off = True
            if terminated and reward < -1.0:  # Collision gives -2.0
                collided = True
            
            done = terminated or truncated
            
            if render:
                time.sleep(0.03)
        
        # Determine outcome
        if dropped_off:
            outcome = "SUCCESS"
        elif collided:
            outcome = "COLLISION"
        else:
            outcome = "TIMEOUT"
        
        # Count action distribution for this episode
        ep_action_dist = np.bincount(ep_actions, minlength=7)
        
        results.append({
            "episode": ep + 1,
            "outcome": outcome,
            "reward": total_reward,
            "steps": steps,
            "picked_up": picked_up,
            "dropped_off": dropped_off,
            "collided": collided,
            "action_dist": ep_action_dist,
        })
        
        print(f"  Ep {ep+1:02d} | {outcome:<9} | Reward: {total_reward:7.2f} | Steps: {steps:3d} | Pickup: {'✓' if picked_up else '✗'} | Dropoff: {'✓' if dropped_off else '✗'}")
    
    env.close()
    
    # Aggregate statistics
    rewards = [r["reward"] for r in results]
    steps_list = [r["steps"] for r in results]
    successes = sum(1 for r in results if r["outcome"] == "SUCCESS")
    collisions = sum(1 for r in results if r["outcome"] == "COLLISION")
    timeouts = sum(1 for r in results if r["outcome"] == "TIMEOUT")
    pickups = sum(1 for r in results if r["picked_up"])
    
    action_names = ["Forward", "Backward", "Turn Left", "Turn Right", "Pick Up", "Drop Off", "Stay"]
    total_actions = action_counts.sum()
    
    print(f"\n  --- Level {difficulty} Summary ---")
    print(f"  Success Rate : {successes}/{num_episodes} ({100*successes/num_episodes:.0f}%)")
    print(f"  Collisions   : {collisions}/{num_episodes}")
    print(f"  Timeouts     : {timeouts}/{num_episodes}")
    print(f"  Pickups      : {pickups}/{num_episodes}")
    print(f"  Avg Reward   : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Avg Steps    : {np.mean(steps_list):.0f}")
    print(f"  Reward Range : [{min(rewards):.2f}, {max(rewards):.2f}]")
    print(f"\n  Action Distribution:")
    for i, name in enumerate(action_names):
        pct = 100 * action_counts[i] / total_actions if total_actions > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {name:<10}: {action_counts[i]:5d} ({pct:5.1f}%) {bar}")
    
    return {
        "difficulty": difficulty,
        "num_episodes": num_episodes,
        "success_rate": successes / num_episodes,
        "collision_rate": collisions / num_episodes,
        "timeout_rate": timeouts / num_episodes,
        "pickup_rate": pickups / num_episodes,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_steps": float(np.mean(steps_list)),
        "action_distribution": {name: int(action_counts[i]) for i, name in enumerate(action_names)},
        "results": results,
    }

def main():
    # Find the best checkpoint
    checkpoint_dir = "models/checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoints = sorted(
            [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")],
            key=lambda x: int(x.split("_")[-2]) if x.split("_")[-2].isdigit() else 0
        )
        if checkpoints:
            best_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
        else:
            best_checkpoint = None
    else:
        best_checkpoint = None
    
    # Also check for final model
    final_model = "models/recurrent_ppo_hivemind_final.zip"
    
    if os.path.exists(final_model):
        model_path = final_model
    elif best_checkpoint and os.path.exists(best_checkpoint):
        model_path = best_checkpoint
    else:
        print("ERROR: No RecurrentPPO model found!")
        print(f"  Checked: {final_model}")
        print(f"  Checked: {checkpoint_dir}/")
        sys.exit(1)
    
    print("=" * 60)
    print("  RecurrentPPO (LSTM) Model Evaluation")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Model Size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
    
    # Test across all difficulty levels
    all_results = {}
    for level in [1, 2, 3, 4]:
        all_results[level] = run_evaluation(model_path, difficulty=level, num_episodes=10, render=False)
    
    # Print final cross-level summary
    print(f"\n{'='*60}")
    print(f"  CROSS-LEVEL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Level':<7} {'Success':<10} {'Collision':<11} {'Timeout':<9} {'Pickup':<8} {'Avg Reward':<12}")
    print(f"  {'-'*57}")
    for level in [1, 2, 3, 4]:
        r = all_results[level]
        print(f"  {level:<7} {r['success_rate']*100:5.0f}%     {r['collision_rate']*100:5.0f}%      {r['timeout_rate']*100:5.0f}%    {r['pickup_rate']*100:5.0f}%   {r['avg_reward']:8.2f}")
    
    print(f"\n  DONE. Full evaluation complete.")

if __name__ == "__main__":
    main()
