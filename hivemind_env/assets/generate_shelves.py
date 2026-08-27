import os

def generate_shelf_urdf(length_m, filepath):
    # Base and shelves
    urdf = f"""<?xml version="1.0"?>
<robot name="shelf_{length_m}m">

  <material name="brown">
    <color rgba="0.55 0.27 0.07 1.0"/>
  </material>

  <material name="black">
    <color rgba="0.1 0.1 0.1 1.0"/>
  </material>

  <link name="base"/>

"""
    
    # Add 3 shelves
    heights = [0.3, 1.0, 1.8]
    for i, h in enumerate(heights):
        urdf += f"""
  <link name="shelf_{i+1}">
    <visual>
      <geometry><box size="{length_m}.0 1.0 0.08"/></geometry>
      <material name="brown"/>
    </visual>
    <collision>
      <geometry><box size="{length_m}.0 1.0 0.08"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="{15.0 * (length_m / 3.0)}"/>
      <inertia ixx="1.258" ixy="0" ixz="0"
               iyy="11.258" iyz="0" izz="12.5"/>
    </inertial>
  </link>

  <joint name="shelf_{i+1}_joint" type="fixed">
    <parent link="base"/>
    <child link="shelf_{i+1}"/>
    <origin xyz="0 0 {h}"/>
  </joint>
"""

    # Add posts every 1 meter
    post_x_positions = [-length_m/2.0 + i for i in range(length_m + 1)]
    post_idx = 1
    for px in post_x_positions:
        for py, pos_name in [(-0.5, "front"), (0.5, "rear")]:
            urdf += f"""
  <link name="post_{pos_name}_{post_idx}">
    <visual>
      <geometry><box size="0.06 0.06 2.0"/></geometry>
      <material name="black"/>
    </visual>
    <collision>
      <geometry><box size="0.06 0.06 2.0"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="3.0"/>
      <inertia ixx="1.0009" ixy="0" ixz="0"
               iyy="1.0009" iyz="0" izz="0.0018"/>
    </inertial>
  </link>

  <joint name="post_{pos_name}_{post_idx}_joint" type="fixed">
    <parent link="base"/>
    <child link="post_{pos_name}_{post_idx}"/>
    <origin xyz="{px} {py} 1.0"/>
  </joint>
"""
        post_idx += 1

    urdf += "</robot>\n"
    
    with open(filepath, "w") as f:
        f.write(urdf)

if __name__ == "__main__":
    assets_dir = os.path.dirname(os.path.abspath(__file__))
    for l in range(1, 8):
        generate_shelf_urdf(l, os.path.join(assets_dir, f"shelf_{l}m.urdf"))
    print("Successfully generated shelf URDFs for lengths 1m to 7m.")
