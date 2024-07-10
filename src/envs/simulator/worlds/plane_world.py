"""Build a simple world with plane only."""


class PlaneWorld:
    """Builds a simple world with a plane only."""

    def __init__(self, pybullet_client, search_path="src/envs/simulator/sim_envs_v2"):
        self._pybullet_client = pybullet_client
        self._pybullet_client.setAdditionalSearchPath(search_path)

    def build_world(self, friction=.99):
        """Builds world with a simple plane and custom friction."""
        ground_id = self._pybullet_client.loadURDF('urdf/plane.urdf')
        # self._pybullet_client.changeDynamics(ground_id, -1, lateralFriction=.99)
        self._pybullet_client.changeDynamics(ground_id, -1, lateralFriction=friction)
        return ground_id
