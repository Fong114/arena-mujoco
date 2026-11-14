"""Per-joint position servo gains sized from URDF limits, the rule arena_isaac's drives apply."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

GRAVITY = 9.81
STIFFNESS_PER_EFFORT_REVOLUTE = 100.0
STIFFNESS_PER_EFFORT_PRISMATIC = 8.0
MAX_GRAVITY_SAG = 0.002
STIFFNESS_DAMPING_RATIO = 20.0
EFFORT_FALLBACK = 25.0


def _carried_mass(root: ET.Element, joint: ET.Element) -> float:
    children: dict[str, list[str]] = {}
    for other in root.findall('joint'):
        parent, child = other.find('parent'), other.find('child')
        if parent is not None and child is not None:
            children.setdefault(parent.get('link', ''), []).append(child.get('link', ''))
    masses = {}
    for link in root.findall('link'):
        mass = link.find('inertial/mass')
        masses[link.get('name', '')] = float(mass.get('value', 0.0)) if mass is not None else 0.0
    child = joint.find('child')
    stack = [child.get('link', '')] if child is not None else []
    seen: set[str] = set()
    total = 0.0
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        total += masses.get(name, 0.0)
        stack.extend(children.get(name, []))
    return total


def position_gains(root: ET.Element, joint: ET.Element) -> tuple[float, float] | None:
    """Stiffness and damping of a position-commanded joint, None for a joint type without a servo."""
    kind = joint.get('type')
    if kind not in ('revolute', 'continuous', 'prismatic'):
        return None
    limit = joint.find('limit')
    effort = float(limit.get('effort') or EFFORT_FALLBACK) if limit is not None else EFFORT_FALLBACK
    velocity = float(limit.get('velocity', 0.0)) if limit is not None else 0.0
    if kind == 'prismatic':
        mass = _carried_mass(root, joint)
        stiffness = max(STIFFNESS_PER_EFFORT_PRISMATIC * effort, mass * GRAVITY / MAX_GRAVITY_SAG)
        damping = 2.0 * math.sqrt(stiffness * mass) if mass > 0.0 else stiffness / STIFFNESS_DAMPING_RATIO
    else:
        stiffness = STIFFNESS_PER_EFFORT_REVOLUTE * effort
        damping = stiffness / STIFFNESS_DAMPING_RATIO
    if velocity > 0.0:
        damping = min(damping, effort / velocity)
    return stiffness, damping


def position_drives(urdf_path: str) -> dict[str, tuple[float, float]]:
    """Stiffness and damping per joint that ros2_control commands by position, keyed like the command interfaces."""
    root = ET.parse(urdf_path).getroot()
    joints = {joint.get('name'): joint for joint in root.findall('joint')}
    drives: dict[str, tuple[float, float]] = {}
    for rc in root.iter('ros2_control'):
        for entry in rc.findall('joint'):
            name = entry.get('name')
            if not name or name not in joints or not any(ci.get('name') == 'position' for ci in entry.findall('command_interface')):
                continue
            if (gains := position_gains(root, joints[name])) is not None:
                drives[name.replace('-', '_')] = gains
    return drives
