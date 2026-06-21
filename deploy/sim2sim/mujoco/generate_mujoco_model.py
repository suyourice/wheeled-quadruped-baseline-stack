from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
import os

import mujoco


LEG_POSITION_ACTUATORS = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)

WHEEL_VELOCITY_ACTUATORS = (
    "FL_foot_joint",
    "FR_foot_joint",
    "RL_foot_joint",
    "RR_foot_joint",
)

LEG_JOINT_DAMPING = "2.0"
LEG_JOINT_ARMATURE = "0.01"
WHEEL_JOINT_DAMPING = "0.0"
WHEEL_JOINT_ARMATURE = "0.01"
DEFAULT_LEG_KP = "40"
DEFAULT_WHEEL_KV = "0.5"


def _patch_mesh_path(filename: str, asset_root: Path) -> str:
    if filename.startswith("package://go2w/"):
        relative = filename.replace("package://go2w/", "")
        return str(asset_root / relative)

    if filename.startswith("package://go2w_description/"):
        relative = filename.replace("package://go2w_description/", "")
        return str(asset_root / relative)

    return filename


def _remove_material_tags(root: ET.Element) -> None:
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "material":
                parent.remove(child)


def _add_floating_base_joint(root: ET.Element) -> None:
    if root.find("./link[@name='world']") is None:
        root.insert(0, ET.Element("link", {"name": "world"}))

    if root.find("./joint[@name='floating_base']") is None:
        floating_joint = ET.Element("joint", {"name": "floating_base", "type": "floating"})

        parent = ET.SubElement(floating_joint, "parent")
        parent.set("link", "world")

        child = ET.SubElement(floating_joint, "child")
        child.set("link", "base")

        origin = ET.SubElement(floating_joint, "origin")
        origin.set("xyz", "0 0 0.35")
        origin.set("rpy", "0 0 0")

        root.insert(1, floating_joint)


def create_mujoco_compatible_urdf(src_urdf: Path, dst_urdf: Path, asset_root: Path) -> None:
    tree = ET.parse(src_urdf)
    root = tree.getroot()

    _remove_material_tags(root)
    _add_floating_base_joint(root)

    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if filename:
            mesh.attrib["filename"] = _patch_mesh_path(filename, asset_root)

    dst_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_urdf, encoding="utf-8", xml_declaration=True)


def _ensure_worldbody(root: ET.Element) -> ET.Element:
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    return worldbody


def _insert_floor(root: ET.Element) -> None:
    worldbody = _ensure_worldbody(root)

    existing_floor = worldbody.find("./geom[@name='floor']")
    if existing_floor is not None:
        worldbody.remove(existing_floor)

    floor = ET.Element(
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": "0 0 0",
            "size": "20 20 0.05",
            "friction": os.environ.get("GO2W_MJ_FLOOR_FRICTION", "0.5 0.005 0.0001"),
        },
    )

    worldbody.insert(0, floor)


def _patch_joint_dynamics(root: ET.Element) -> None:
    leg_joint_names = set(LEG_POSITION_ACTUATORS)
    wheel_joint_names = set(WHEEL_VELOCITY_ACTUATORS)

    for joint in root.iter("joint"):
        name = joint.attrib.get("name", "")

        if name in leg_joint_names:
            joint.set("damping", os.environ.get("GO2W_MJ_LEG_DAMPING", LEG_JOINT_DAMPING))
            joint.set("armature", LEG_JOINT_ARMATURE)

        elif name in wheel_joint_names:
            joint.set("damping", os.environ.get("GO2W_MJ_WHEEL_DAMPING", WHEEL_JOINT_DAMPING))
            joint.set("armature", WHEEL_JOINT_ARMATURE)


def _patch_joint_force_limits(root: ET.Element) -> None:
    leg_joint_names = set(LEG_POSITION_ACTUATORS)
    wheel_joint_names = set(WHEEL_VELOCITY_ACTUATORS)

    leg_effort = float(os.environ.get("GO2W_MJ_LEG_EFFORT_LIMIT", "40.0"))
    wheel_effort = float(os.environ.get("GO2W_MJ_WHEEL_EFFORT_LIMIT", "23.7"))

    patched_leg_count = 0
    patched_wheel_count = 0

    for joint in root.iter("joint"):
        name = joint.attrib.get("name", "")

        if name in leg_joint_names:
            joint.set("actuatorfrcrange", f"{-leg_effort:g} {leg_effort:g}")
            patched_leg_count += 1

        elif name in wheel_joint_names:
            joint.set("actuatorfrcrange", f"{-wheel_effort:g} {wheel_effort:g}")
            patched_wheel_count += 1

    print(
        "Patched joint force limits: "
        f"legs={patched_leg_count} at +/-{leg_effort:g}, "
        f"wheels={patched_wheel_count} at +/-{wheel_effort:g}"
    )


def _patch_wheel_contact(root: ET.Element) -> None:
    wheel_body_names = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"}
    wheel_friction = os.environ.get("GO2W_MJ_WHEEL_FRICTION", "0.5 0.005 0.0001")
    wheel_condim = os.environ.get("GO2W_MJ_WHEEL_CONDIM", None)

    patched_count = 0
    for body in root.findall(".//body"):
        if body.get("name") not in wheel_body_names:
            continue

        for geom in body.findall("geom"):
            geom.set("friction", wheel_friction)
            if wheel_condim is not None:
                geom.set("condim", wheel_condim)
            patched_count += 1

    print(f"Patched wheel contact geoms: {patched_count}, friction={wheel_friction}")


def _insert_actuators(root: ET.Element) -> None:
    old_actuator = root.find("actuator")
    if old_actuator is not None:
        root.remove(old_actuator)

    actuator = ET.Element("actuator")

    for joint_name in LEG_POSITION_ACTUATORS:
        ET.SubElement(
            actuator,
            "position",
            {
                "name": joint_name.replace("_joint", "_pos"),
                "joint": joint_name,
                "kp": os.environ.get("GO2W_MJ_LEG_KP", DEFAULT_LEG_KP),
                "ctrlrange": "-2.7 2.7",
                "ctrllimited": "true",
            },
        )

    for joint_name in WHEEL_VELOCITY_ACTUATORS:
        ET.SubElement(
            actuator,
            "velocity",
            {
                "name": joint_name.replace("_foot_joint", "_wheel_vel"),
                "joint": joint_name,
                "kv": os.environ.get("GO2W_MJ_WHEEL_KV", DEFAULT_WHEEL_KV),
                "ctrlrange": "-30.1 30.1",
                "ctrllimited": "true",
            },
        )

    root.append(actuator)


def patch_mjcf(base_mjcf: Path, output_path: Path) -> None:
    tree = ET.parse(base_mjcf)
    root = tree.getroot()

    _insert_floor(root)
    _patch_joint_dynamics(root)

    _patch_joint_force_limits(root)

    _patch_wheel_contact(root)

    print(
        "MuJoCo actuator overrides: "
        f"GO2W_MJ_LEG_KP={os.environ.get('GO2W_MJ_LEG_KP', DEFAULT_LEG_KP)}, "
        f"GO2W_MJ_LEG_DAMPING={os.environ.get('GO2W_MJ_LEG_DAMPING', LEG_JOINT_DAMPING)}, "
        f"GO2W_MJ_WHEEL_DAMPING={os.environ.get('GO2W_MJ_WHEEL_DAMPING', WHEEL_JOINT_DAMPING)}, "
        f"GO2W_MJ_WHEEL_KV={os.environ.get('GO2W_MJ_WHEEL_KV', DEFAULT_WHEEL_KV)}"
    )

    if os.environ.get("GO2W_DISABLE_NON_FOOT_LEG_COLLISIONS", "0") == "1":
        _disable_non_foot_leg_collisions(root)
    else:
        print("Non-foot leg collision geoms: enabled")
    _insert_actuators(root)

    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def _disable_non_foot_leg_collisions(root: ET.Element) -> None:
    disabled_bodies = {
        "FL_hip",
        "FL_thigh",
        "FL_calf",
        "FR_hip",
        "FR_thigh",
        "FR_calf",
        "RL_hip",
        "RL_thigh",
        "RL_calf",
        "RR_hip",
        "RR_thigh",
        "RR_calf",
    }

    disabled_count = 0
    for body in root.findall(".//body"):
        body_name = body.get("name")
        if body_name not in disabled_bodies:
            continue

        for geom in body.findall("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            disabled_count += 1

    print(f"Disabled non-foot leg collision geoms: {disabled_count}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    asset_root = repo_root / "assets" / "go2w"
    src_urdf = asset_root / "urdf" / "go2w.urdf"

    generated_dir = Path(__file__).resolve().parent / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    cleaned_urdf = generated_dir / "go2w_mujoco_floating.urdf"
    base_mjcf = generated_dir / "go2w_mujoco_base.xml"
    actuated_mjcf = generated_dir / "go2w_mujoco_actuated.xml"

    print(f"Source URDF: {src_urdf}")
    print(f"Cleaned URDF: {cleaned_urdf}")
    print(f"Base MJCF: {base_mjcf}")
    print(f"Actuated MJCF: {actuated_mjcf}")

    create_mujoco_compatible_urdf(src_urdf, cleaned_urdf, asset_root)

    model = mujoco.MjModel.from_xml_path(str(cleaned_urdf))
    mujoco.mj_saveLastXML(str(base_mjcf), model)

    patch_mjcf(base_mjcf, actuated_mjcf)

    actuated_model = mujoco.MjModel.from_xml_path(str(actuated_mjcf))

    print("Generated actuated MuJoCo model successfully.")
    print(f"nq={actuated_model.nq}, nv={actuated_model.nv}, nu={actuated_model.nu}")
    print(f"nbody={actuated_model.nbody}, njnt={actuated_model.njnt}, ngeom={actuated_model.ngeom}")

    print("\nActuators:")
    for actuator_id in range(actuated_model.nu):
        name = mujoco.mj_id2name(
            actuated_model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )
        trnid = actuated_model.actuator_trnid[actuator_id]
        print(f"  {actuator_id:02d}: name={name}, trnid={trnid.tolist()}")


if __name__ == "__main__":
    main()
