import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sb3_contrib import RecurrentPPO

from hivemind_env.training import INFERENCE_CUSTOM_OBJECTS

model = RecurrentPPO.load(
    "models/ppo_recurrent_final.zip",
    device="cpu",
    custom_objects=INFERENCE_CUSTOM_OBJECTS,
)
print("Action space:", model.action_space)
print("Observation space:", model.observation_space)
