"""Joint servo gains and effort limits read from the served URDF's ros2_control joint params."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass(frozen=True)
class JointDrive:
    kp: float
    kd: float
    effort: float
    kind: str


def joint_drives(urdf_path: str) -> dict[str, JointDrive]:
    """Drive of every ros2_control joint with a position or effort command interface and kp, kd and effort params."""
    drives: dict[str, JointDrive] = {}
    for rc in ET.parse(urdf_path).getroot().iter('ros2_control'):
        for entry in rc.findall('joint'):
            name = entry.get('name')
            interfaces = {ci.get('name') for ci in entry.findall('command_interface')}
            kind = 'effort' if 'effort' in interfaces else 'position' if 'position' in interfaces else None
            params = {p.get('name'): (p.text or '').strip() for p in entry.findall('param')}
            if not name or kind is None or not {'kp', 'kd', 'effort'} <= params.keys():
                continue
            drives[name.replace('-', '_')] = JointDrive(float(params['kp']), float(params['kd']), float(params['effort']), kind)
    return drives
