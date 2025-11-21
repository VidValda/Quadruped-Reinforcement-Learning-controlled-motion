import os
from pathlib import Path

import mujoco
from robot_descriptions import spot_mj_description
from robot_descriptions.loaders.mujoco import load_robot_description


class SpotModelLoader:

    FLOOR_ASSET = """
        <asset>
            <texture type="2d" name="grid" builtin="checker" width="512" height="512"
                     rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"/>
            <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
        </asset>
    """

    FLOOR_GEOM = '    <geom name="floor" type="plane" size="10 10 0.1" material="grid"/>'

    def __init__(self, assets_subdir: str = "assets") -> None:
        self.assets_subdir = assets_subdir

    def build(self) -> mujoco.MjModel:
        xml_string = self._load_mjcf()
        xml_string = self._ensure_mesh_paths(xml_string)
        xml_string = self._ensure_floor(xml_string)

        try:
            return mujoco.MjModel.from_xml_string(xml_string)
        except (mujoco.Error, RuntimeError, ValueError) as exc:
            print(f"Error compiling XML: {exc}. Loading default.")
            return load_robot_description("spot_mj_description")

    @staticmethod
    def _load_mjcf() -> str:
        xml_path = Path(spot_mj_description.MJCF_PATH)
        with xml_path.open("r", encoding="utf-8") as file:
            return file.read()

    def _ensure_mesh_paths(self, xml_string: str) -> str:
        xml_path = Path(spot_mj_description.MJCF_PATH)
        xml_dir = xml_path.parent
        assets_dir = os.path.abspath(xml_dir / self.assets_subdir).replace("\\", "/")

        if 'meshdir="assets"' in xml_string:
            return xml_string.replace('meshdir="assets"', f'meshdir="{assets_dir}"')

        if "<compiler" not in xml_string:
            return xml_string.replace(
                '<mujoco model="spot">',
                f'<mujoco model="spot">\n  <compiler meshdir="{assets_dir}"/>',
                1,
            )

        return xml_string.replace(
            "<compiler",
            f'<compiler meshdir="{assets_dir}"',
            1,
        )

    def _ensure_floor(self, xml_string: str) -> str:
        if "<asset>" not in xml_string:
            if "</compiler>" in xml_string:
                xml_string = xml_string.replace("</compiler>", f"</compiler>\n{self.FLOOR_ASSET}", 1)
            else:
                xml_string = xml_string.replace('<mujoco model="spot">', f'<mujoco model="spot">\n{self.FLOOR_ASSET}', 1)
        else:
            asset_additions = """
            <texture type="2d" name="grid" builtin="checker" width="512" height="512" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"/>
            <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2"/>
            """
            xml_string = xml_string.replace("<asset>", f"<asset>\n{asset_additions}", 1)

        if "<worldbody>" in xml_string:
            return xml_string.replace("<worldbody>", f"<worldbody>\n{self.FLOOR_GEOM}", 1)

        print("Warning: <worldbody> not found. Floor not added.")
        return xml_string

