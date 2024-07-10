"""utils for loading environment"""
import pybullet
from pybullet_utils import bullet_client


def add_terrain(p: bullet_client.BulletClient):
    boxHalfLength = 0.2
    boxHalfWidth = 2.5
    boxHalfHeight = 0.05
    sh_colBox = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, boxHalfWidth, 0.05])
    sh_final_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, boxHalfWidth, 0.05])
    boxOrigin = 0.8 + boxHalfLength
    step1 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_colBox,
                              basePosition=[boxOrigin, 1, boxHalfHeight],
                              baseOrientation=[0.0, 0.0, 0.0, 1])

    step2 = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sh_final_col,
                              basePosition=[boxOrigin + 0.5 + boxHalfLength, 1, 0.05 + 2 * boxHalfHeight],
                              baseOrientation=[0.0, 0.0, 0.0, 1])

    p.changeDynamics(step1, -1, lateralFriction=0.85)
    p.changeDynamics(step2, -1, lateralFriction=0.85)


def add_lane(p: bullet_client.BulletClient):
    # all units are in meters
    track_length = 15
    track_width = 0.03
    track_height = 0.0005
    lane_half_width = 0.6
    track_left = p.createVisualShape(p.GEOM_BOX, halfExtents=[track_length, track_width, track_height],
                                     rgbaColor=[1, 0, 0, 0.7])
    track_middle = p.createVisualShape(p.GEOM_BOX, halfExtents=[track_length, track_width, track_height],
                                       rgbaColor=[0, 0, 1, 0.7])
    track_right = p.createVisualShape(p.GEOM_BOX, halfExtents=[track_length, track_width, track_height],
                                      rgbaColor=[1, 0, 0, 0.7])

    boxOrigin_x = 0
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=track_left,
                      basePosition=[boxOrigin_x, lane_half_width, 0.0005],
                      baseOrientation=[0.0, 0.0, 0.0, 1])

    p.createMultiBody(baseMass=0, baseVisualShapeIndex=track_middle,
                      basePosition=[boxOrigin_x, 0, 0.0005],
                      baseOrientation=[0.0, 0.0, 0.0, 1])

    p.createMultiBody(baseMass=0, baseVisualShapeIndex=track_right,
                      basePosition=[boxOrigin_x, -lane_half_width, 0.0005],
                      baseOrientation=[0.0, 0.0, 0.0, 1])