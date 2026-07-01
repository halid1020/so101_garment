"""Module-level helpers for the robot visualizer."""

from typing import Any

import numpy as np
import viser
import yourdfpy
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf


class RobotVisualizerCore:
    """
    Handles the 3D rendering of the robot, ghost robot, and target frames.

    This class encapsulates the Viser server and provides a structured interface
    for updating the visual state of the main robot, a semi-transparent 'ghost'
    robot used for previews or targets, UI camera streams, and spatial controllers.
    """

    def __init__(self, urdf_path: str) -> None:
        """
        Initializes the visualization server and loads the robot URDFs.

        Args:
            urdf_path (str): The file path to the robot's URDF model.
        """
        self.server = viser.ViserServer()
        self.server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        # Load actual robot URDF
        urdf = yourdfpy.URDF.load(urdf_path)
        self.urdf_vis = ViserUrdf(self.server, urdf, root_node_name="/robot_actual")

        # Load ghost robot URDF
        ghost_urdf = yourdfpy.URDF.load(urdf_path)
        self.ghost_robot_urdf = ViserUrdf(
            self.server,
            ghost_urdf,
            root_node_name="/robot_ghost",
            mesh_color_override=(1.0, 0.65, 0.0, 0.25),
        )

        self.controller_handle: Any = None
        self.target_frame_handle: Any = None
        self.rgb_image_handle: Any = None

    def add_controller_visualization(self) -> None:
        """
        Adds an interactive 3D transform control widget to the scene.

        This allows users to drag a 3D handle in the web UI, which can be
        linked back to teleoperation or inverse kinematics targets.
        """
        self.controller_handle = self.server.scene.add_transform_controls(
            "/controller", scale=0.15, position=(0, 0, 0), wxyz=(1, 0, 0, 0)
        )

    def add_target_frame_visualization(self) -> None:
        """
        Adds a static 3D coordinate frame representing a target or goal position.
        """
        self.target_frame_handle = self.server.scene.add_frame(
            "/target_goal", axes_length=0.1, axes_radius=0.003
        )

    def add_rgb_image_placeholder(self, height: int = 480, width: int = 640) -> None:
        """
        Creates a 2D image panel in the Viser GUI to display camera streams.

        Args:
            height (int, optional): The height of the placeholder image. Defaults to 480.
            width (int, optional): The width of the placeholder image. Defaults to 640.
        """
        if self.rgb_image_handle is None:
            dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
            self.rgb_image_handle = self.server.gui.add_image(
                dummy_image, label="RGB Camera", format="jpeg", jpeg_quality=85
            )

    def update_rgb_image(self, rgb_image: np.ndarray | None) -> None:
        """
        Updates the RGB camera panel in the GUI with a new image frame.

        Args:
            rgb_image (np.ndarray | None): A numpy array representing the image,
                                           or None to skip updating.
        """
        if rgb_image is None:
            return
        if self.rgb_image_handle is None:
            self.add_rgb_image_placeholder(
                height=rgb_image.shape[0], width=rgb_image.shape[1]
            )
        # Type ignored because mypy cannot infer that the handle is initialized
        self.rgb_image_handle.image = rgb_image  # type: ignore

    def update_robot_pose(self, joint_config: np.ndarray) -> None:
        """
        Updates the primary robot's joint configurations to render its current physical pose.

        Args:
            joint_config (np.ndarray): An array of joint positions in radians.
        """
        self.urdf_vis.update_cfg(joint_config)

    def update_ghost_robot_pose(self, joint_config: np.ndarray) -> None:
        """
        Updates the ghost robot's joint configurations to reflect simulated or target poses.

        Args:
            joint_config (np.ndarray): An array of joint positions in radians.
        """
        if self.ghost_robot_urdf:
            self.ghost_robot_urdf.update_cfg(joint_config)

    def update_ghost_robot_visibility(self, flag: bool) -> None:
        """
        Toggles the visibility of the ghost robot in the scene.

        Args:
            flag (bool): True to render the ghost robot, False to hide it.
        """
        if self.ghost_robot_urdf:
            self.ghost_robot_urdf.show_visual = flag

    def get_ghost_robot_visibility(self) -> bool:
        """
        Retrieves the current visibility state of the ghost robot.

        Returns:
            bool: True if the ghost robot is currently visible, False otherwise.
        """
        if self.ghost_robot_urdf:
            return self.ghost_robot_urdf.show_visual
        return False

    def set_ghost_robot_color(self, color: tuple[float, float, float, float]) -> None:
        """
        Overrides the mesh color of the ghost robot.

        Args:
            color (tuple[float, float, float, float]): An RGBA tuple representing
                                                       the desired color.
        """
        if self.ghost_robot_urdf:
            self.ghost_robot_urdf.mesh_color_override = color

    def update_controller_visualization(self, transform: np.ndarray | None) -> None:
        """
        Updates the physical location and rotation of the interactive 3D controller.

        Args:
            transform (np.ndarray | None): A 4x4 homogenous transformation matrix.
        """
        if self.controller_handle is None or transform is None:
            return
        pos = transform[:3, 3]
        rot = Rotation.from_matrix(transform[:3, :3]).as_quat()
        self.controller_handle.position = tuple(pos)
        self.controller_handle.wxyz = (rot[3], rot[0], rot[1], rot[2])

    def update_target_visualization(self, transform: np.ndarray | None) -> None:
        """
        Updates the physical location and rotation of the target goal frame.

        Args:
            transform (np.ndarray | None): A 4x4 homogenous transformation matrix.
        """
        if self.target_frame_handle is None or transform is None:
            return
        pos = transform[:3, 3]
        rot = Rotation.from_matrix(transform[:3, :3]).as_quat()
        self.target_frame_handle.position = tuple(pos)
        self.target_frame_handle.wxyz = (rot[3], rot[0], rot[1], rot[2])

    def stop(self) -> None:
        """
        Safely stops the Viser web server and releases background resources.
        """
        self.server.stop()
