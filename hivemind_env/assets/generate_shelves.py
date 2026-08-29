import copy
import os
import xml.etree.ElementTree as ET

# Shelf plate heights, in metres, bottom-up. Plates are 0.08 thick, so the bottom one
# spans SHELF_HEIGHTS[0] +/- 0.04.
#
# The bottom plate was at 0.30 (spanning 0.26 - 0.34) until 2026-08-29. The robot
# chassis tops out at 0.194 and the wheels at 0.210, so nothing ever touched: robots
# drove straight under the shelving and aisles did not constrain routing at all. The
# corner posts do reach the floor, but they sit at 1 m spacing on the cell corners,
# which is exactly where a grid-centred robot is not.
#
# Lowering the bottom plate to 0.18 makes it span 0.14 - 0.22, overlapping both the
# chassis and the wheels by ~5 cm, so a robot entering a shelf cell generates real
# contacts and pays the collision penalty. Cartons stocked on that plate move down with
# it - add_cartons() reads the same list, so the two cannot drift apart.
SHELF_HEIGHTS = [0.18, 1.0, 1.8]
CARTON_SHELF_HEIGHTS = SHELF_HEIGHTS[:2]  # top shelf is left empty


def add_cartons(urdf, length_m, carton_filepath):
  carton_root = ET.parse(carton_filepath).getroot()

  for material in carton_root.findall("material"):
    urdf += ET.tostring(material, encoding="unicode")

  shelf_heights = CARTON_SHELF_HEIGHTS
  bay_centers = [-length_m / 2.0 + 0.5 + bay for bay in range(length_m)]
  for shelf_index, shelf_height in enumerate(shelf_heights, start=1):
    for bay_index, bay_center in enumerate(bay_centers, start=1):
      name_prefix = f"carton_{shelf_index}_{bay_index}"
      carton_links = {}

      for link in carton_root.findall("link"):
        renamed_link = copy.deepcopy(link)
        old_name = link.get("name")
        new_name = f"{name_prefix}_{old_name}"
        renamed_link.set("name", new_name)
        if renamed_link.find("inertial") is None:
          inertial = ET.SubElement(renamed_link, "inertial")
          ET.SubElement(inertial, "mass", {"value": "0"})
          ET.SubElement(inertial, "inertia", {
            "ixx": "0", "ixy": "0", "ixz": "0",
            "iyy": "0", "iyz": "0", "izz": "0",
          })
        carton_links[old_name] = new_name
        urdf += ET.tostring(renamed_link, encoding="unicode")

      attachment = ET.Element("joint", {
        "name": f"{name_prefix}_mount_joint",
        "type": "fixed",
      })
      ET.SubElement(attachment, "parent", {"link": "base"})
      ET.SubElement(attachment, "child", {"link": carton_links["carton"]})
      ET.SubElement(attachment, "origin", {
        "xyz": f"{bay_center:.4f} 0 {shelf_height + 0.04:.4f}"
      })
      urdf += ET.tostring(attachment, encoding="unicode")

      for joint in carton_root.findall("joint"):
        renamed_joint = copy.deepcopy(joint)
        renamed_joint.set("name", f"{name_prefix}_{joint.get('name')}")
        parent = renamed_joint.find("parent")
        child = renamed_joint.find("child")
        parent.set("link", carton_links[parent.get("link")])
        child.set("link", carton_links[child.get("link")])
        urdf += ET.tostring(renamed_joint, encoding="unicode")

  return urdf


def generate_shelf_urdf(length_m, filepath, carton_filepath):
    # Base and shelves
    urdf = f"""<?xml version="1.0"?>
<robot name="shelf_{length_m}m">

  <material name="brown">
    <color rgba="0.55 0.27 0.07 1.0"/>
  </material>

  <material name="black">
    <color rgba="0.1 0.1 0.1 1.0"/>
  </material>

  <link name="base">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0"
               iyy="0" iyz="0" izz="0"/>
    </inertial>
  </link>

"""
    
    # Add 3 shelves
    heights = SHELF_HEIGHTS
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

    urdf = add_cartons(urdf, length_m, carton_filepath)
    urdf += "</robot>\n"
    
    with open(filepath, "w") as f:
        f.write(urdf)

if __name__ == "__main__":
    assets_dir = os.path.dirname(os.path.abspath(__file__))
    carton_filepath = os.path.join(assets_dir, "carton.urdf")
    for l in range(1, 8):
      generate_shelf_urdf(l, os.path.join(assets_dir, f"shelf_{l}m.urdf"), carton_filepath)
    print("Successfully generated shelf URDFs for lengths 1m to 7m.")
