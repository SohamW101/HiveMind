import os
import sys
import numpy as np
from stable_baselines3 import PPO

# Add current directory to sys.path
sys.path.insert(0, os.path.dirname(__file__))
from hivemind_env.env import HiveMindSingleAgentEnv

def run_stat_evaluation(model_path="models/ppo_hivemind_v1_final.zip", num_episodes_per_level=10):
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return

    model = PPO.load(model_path, device="cpu")
    
    level_descriptions = {
        1: "1 — fixed, no obstacles",
        2: "2 — random spawns",
        3: "3 — random + obstacles",
        4: "4 — random + obstacles"
    }
    
    level_stats = {}
    total_successful = 0
    total_episodes = 4 * num_episodes_per_level

    for level in range(1, 5):
        env = HiveMindSingleAgentEnv(render_mode=None, difficulty_level=level, obs_size=15)
        
        success_cnt = 0
        collision_cnt = 0
        timeout_cnt = 0
        rewards = []
        steps_list = []
        
        for ep in range(num_episodes_per_level):
            obs, info = env.reset(seed=100 * level + ep)
            done = False
            ep_reward = 0.0
            ep_steps = 0
            dropped_off = False
            collided = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_reward += reward
                ep_steps += 1
                
                if reward >= 9.0:
                    dropped_off = True
                if terminated and reward < -1.0:
                    collided = True
                    
                done = terminated or truncated
                
            if dropped_off:
                success_cnt += 1
            elif collided:
                collision_cnt += 1
            else:
                timeout_cnt += 1
                
            rewards.append(ep_reward)
            steps_list.append(ep_steps)
            
        env.close()
        
        total_successful += success_cnt
        success_pct = int(round((success_cnt / num_episodes_per_level) * 100))
        collision_pct = int(round((collision_cnt / num_episodes_per_level) * 100))
        timeout_pct = int(round((timeout_cnt / num_episodes_per_level) * 100))
        avg_reward = np.mean(rewards)
        avg_steps = int(round(np.mean(steps_list)))
        
        level_stats[level] = {
            "desc": level_descriptions[level],
            "success_pct": f"{success_pct}%",
            "collision_pct": f"{collision_pct}%",
            "timeout_pct": f"{timeout_pct}%",
            "avg_reward": f"{avg_reward:.2f}",
            "avg_steps": f"{avg_steps}"
        }

    overall_success_rate = (total_successful / total_episodes) * 100.0
    
    # Print exact heading structure
    print("STATISTICS — EVALUATION")
    print(f"{overall_success_rate:.1f}% success across {total_episodes} evaluation episodes\n")
    
    # Print formatted markdown table matching the image format
    print(f"| {'Level':<25} | {'Success':<8} | {'Collision':<9} | {'Timeout':<8} | {'Avg reward':<10} | {'Avg steps':<9} |")
    print(f"|{'-'*27}|{'-'*10}|{'-'*11}|{'-'*10}|{'-'*12}|{'-'*11}|")
    for lvl in range(1, 5):
        s = level_stats[lvl]
        print(f"| {s['desc']:<25} | {s['success_pct']:^8} | {s['collision_pct']:^9} | {s['timeout_pct']:^8} | {s['avg_reward']:^10} | {s['avg_steps']:^9} |")

if __name__ == "__main__":
    run_stat_evaluation()
