"""Build a simple world with plane only."""


class PlaneWorld:
    """Builds a simple world with a plane only."""

    def __init__(self, pybullet_client):
        self._pybullet_client = pybullet_client
        self._pybullet_client.setAdditionalSearchPath("envs/sim_envs_v2")

    def build_world(self):
        """Builds world with a simple plane and custom friction."""
        ground_id = self._pybullet_client.loadURDF('urdf/plane.urdf')
        # self._pybullet_client.changeDynamics(ground_id, -1, lateralFriction=.99)
        self._pybullet_client.changeDynamics(ground_id, -1, lateralFriction=.7)
        return ground_id
