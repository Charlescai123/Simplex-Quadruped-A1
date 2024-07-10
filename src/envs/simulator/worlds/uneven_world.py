"""Build a world with uneven terrains."""
import numpy as np

class UnevenWorld:
  """Builds a simple world with a plane only."""
  def __init__(self, pybullet_client,  search_path="src/envs/simulator/sim_envs_v2"):
    self._pybullet_client = pybullet_client
    self._pybullet_client.setAdditionalSearchPath(search_path)

  def build_world(self, friction=1.):
    """Builds world with uneven terrains."""
    p = self._pybullet_client
    height_perturbation_range = 0.08
    num_heightfield_rows = 512
    num_heightfield_columns = 512
    heightfield_data = [0] * num_heightfield_rows * num_heightfield_columns
    for j in range(int(num_heightfield_columns / 2)):
      for i in range(int(num_heightfield_rows / 2)):
        height = np.random.uniform(0, height_perturbation_range)
        # height = height_perturbation_range if (i + j) % 2 == 0 else 0
        heightfield_data[2 * i + 2 * j * num_heightfield_rows] = height
        heightfield_data[2 * i + 1 + 2 * j * num_heightfield_rows] = height
        heightfield_data[2 * i + (2 * j + 1) * num_heightfield_rows] = height
        heightfield_data[2 * i + 1 +
                         (2 * j + 1) * num_heightfield_rows] = height

    terrain_shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[.1, .1, 1],  # <-- change this for different granularity
        heightfieldTextureScaling=(num_heightfield_rows - 1) / 2,
        heightfieldData=heightfield_data,
        numHeightfieldRows=num_heightfield_rows,
        numHeightfieldColumns=num_heightfield_columns)
    ground_id = p.createMultiBody(0, terrain_shape)
    # Change friction coefficient to 1 (or anything you want). I think 1 is default,
    # but you can play with this value.
    p.changeDynamics(ground_id, -1, lateralFriction=friction)
    p.changeVisualShape(ground_id, -1, rgbaColor=[1, 1, 0, 1])
    return ground_id
