import time
import math
import pybullet as pb
from hivemind_env.env import HiveMindMultiAgentEnv

def make_path(*corners):
    path = []
    for i in range(len(corners) - 1):
        r1, c1 = corners[i]
        r2, c2 = corners[i+1]
        if r1 == r2:
            step = 1 if c2 > c1 else -1
            for c in range(c1, c2 + step, step):
                path.append((r1, c))
        elif c1 == c2:
            step = 1 if r2 > r1 else -1
            for r in range(r1, r2 + step, step):
                path.append((r, c1))
    clean = [path[0]]
    for p in path[1:]:
        if p != clean[-1]: clean.append(p)
    return clean

def generate_actions(path, start_yaw_dir):
    actions = []
    current_yaw = start_yaw_dir
    
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        
        dr, dc = r2 - r1, c2 - c1
        target_yaw = 0
        if dc == 1: target_yaw = 0
        elif dr == -1: target_yaw = 1
        elif dc == -1: target_yaw = 2
        elif dr == 1: target_yaw = 3
        
        while current_yaw != target_yaw:
            if (current_yaw + 1) % 4 == target_yaw:
                current_yaw = (current_yaw + 1) % 4
                actions.append((2, (r1, c1), current_yaw))
            elif (current_yaw - 1) % 4 == target_yaw:
                current_yaw = (current_yaw - 1) % 4
                actions.append((3, (r1, c1), current_yaw))
            else:
                current_yaw = (current_yaw + 2) % 4
                actions.append((2, (r1, c1), (current_yaw - 1)%4))
                actions.append((2, (r1, c1), current_yaw))
                
        actions.append((0, (r2, c2), current_yaw))
        
    return actions, current_yaw

class Controller:
    def __init__(self, idx, start_pos):
        self.idx = idx
        self.grid_pos = start_pos
        self.yaw_dir = 0
        self.actions = []
        self.step = 0
        self.done = False

    def add_path(self, path):
        new_acts, self.yaw_dir = generate_actions(path, self.yaw_dir)
        self.actions.extend(new_acts)
        
    def add_action(self, action_id):
        self.actions.append((action_id, self.grid_pos, self.yaw_dir))

def build_controllers():
    c0 = Controller(0, (0, 1))
    c1 = Controller(1, (1, 0))
    c2 = Controller(2, (0, 2))
    c3 = Controller(3, (2, 0))
    
    # Bot 0 Ring Road
    c0.add_path(make_path((0,1), (0,0), (2,0), (2,4)))
    c0.add_action(4)
    c0.add_path(make_path((2,4), (2,12), (0,12), (0,1)))
    c0.add_action(5)
    
    c0.add_path(make_path((0,1), (0,0), (6,0), (6,4)))
    c0.add_action(4)
    c0.add_path(make_path((6,4), (6,12), (0,12), (0,1)))
    c0.add_action(5)
    
    c0.add_path(make_path((0,1), (0,0), (10,0), (10,4)))
    c0.add_action(4)
    c0.add_path(make_path((10,4), (10,12), (0,12), (0,1)))
    c0.add_action(5)
    
    c0.add_path(make_path((0,1), (0,0), (12,0), (12,1)))
    
    # Bot 1 Ring Road
    c1.add_path(make_path((1,0), (2,0), (2,8)))
    c1.add_action(4)
    c1.add_path(make_path((2,8), (2,12), (0,12), (0,0), (1,0)))
    c1.add_action(5)
    
    c1.add_path(make_path((1,0), (6,0), (6,8)))
    c1.add_action(4)
    c1.add_path(make_path((6,8), (6,12), (0,12), (0,0), (1,0)))
    c1.add_action(5)
    
    c1.add_path(make_path((1,0), (10,0), (10,8)))
    c1.add_action(4)
    c1.add_path(make_path((10,8), (10,12), (0,12), (0,0), (1,0)))
    c1.add_action(5)
    
    c1.add_path(make_path((1,0), (12,0), (12,2)))
    
    # Bot 2 Ring Road
    c2.add_path(make_path((0,2), (0,0), (4,0), (4,4)))
    c2.add_action(4)
    c2.add_path(make_path((4,4), (4,12), (0,12), (0,1)))
    c2.add_action(5)
    
    c2.add_path(make_path((0,1), (0,0), (8,0), (8,4)))
    c2.add_action(4)
    c2.add_path(make_path((8,4), (8,12), (0,12), (0,1)))
    c2.add_action(5)
    
    c2.add_path(make_path((0,1), (0,0), (12,0), (12,4)))
    c2.add_action(4)
    c2.add_path(make_path((12,4), (12,12), (0,12), (0,1)))
    c2.add_action(5)
    
    c2.add_path(make_path((0,1), (0,0), (12,0), (12,3)))
    
    # Bot 3 Ring Road
    c3.add_path(make_path((2,0), (4,0), (4,8)))
    c3.add_action(4)
    c3.add_path(make_path((4,8), (4,12), (0,12), (0,0), (1,0)))
    c3.add_action(5)
    
    c3.add_path(make_path((1,0), (8,0), (8,8)))
    c3.add_action(4)
    c3.add_path(make_path((8,8), (8,12), (0,12), (0,0), (1,0)))
    c3.add_action(5)
    
    c3.add_path(make_path((1,0), (12,0), (12,8)))
    c3.add_action(4)
    c3.add_path(make_path((12,8), (12,12), (0,12), (0,0), (1,0)))
    c3.add_action(5)
    
    c3.add_path(make_path((1,0), (12,0), (12,4)))
    
    return [c0, c1, c2, c3]

def play_demo():
    print("Initializing 4-Agent Simultaneous Ring Road Choreography...")
    env = HiveMindMultiAgentEnv(render_mode="human")
    obs, info = env.reset()
    
    pb.resetDebugVisualizerCamera(cameraDistance=16.0, cameraYaw=0, cameraPitch=-89.9, cameraTargetPosition=[0, 0, 0])
    
    controllers = build_controllers()
    
    try:
        while True:
            step_actions = []
            planned_pos = []
            
            for i, c in enumerate(controllers):
                if c.step < len(c.actions):
                    a, n_pos, n_yaw = c.actions[c.step]
                    
                    conflict = False
                    for j, other_c in enumerate(controllers):
                        if j != i:
                            # Cannot step into someone's current position
                            if n_pos == other_c.grid_pos:
                                conflict = True
                    # Cannot step into a cell that a higher priority bot just reserved for THIS step
                    if n_pos in planned_pos and n_pos != c.grid_pos:
                        conflict = True
                        
                    if conflict:
                        step_actions.append(6) # Stay
                        planned_pos.append(c.grid_pos)
                    else:
                        step_actions.append(a)
                        planned_pos.append(n_pos)
                else:
                    step_actions.append(6)
                    planned_pos.append(c.grid_pos)
            
            obs, reward, term, trunc, info = env.step(step_actions)
            
            # Commit
            for i, c in enumerate(controllers):
                if step_actions[i] != 6 and c.step < len(c.actions):
                    c.grid_pos = planned_pos[i]
                    c.yaw_dir = c.actions[c.step][2]
                    c.step += 1
            
            if info['remaining_resources'] == 0:
                print("All resources delivered!")
                break
                
            if all(c.step >= len(c.actions) for c in controllers):
                print("All choreographed paths completed.")
                break
                
    except KeyboardInterrupt:
        print("Demo Stopped by User.")
    finally:
        print("Sequence complete. Exiting...")
        time.sleep(3)
        env.close()

if __name__ == "__main__":
    play_demo()
