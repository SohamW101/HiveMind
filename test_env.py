import time
from hivemind_env.env import HiveMindSingleAgentEnv

def test():
    print("Testing environment...")
    env = HiveMindSingleAgentEnv(render_mode="human", difficulty_level=2)
    obs, info = env.reset()
    print("Environment reset successful.")
    
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        time.sleep(0.05)
        
    env.close()
    print("Test finished successfully.")

if __name__ == "__main__":
    test()
