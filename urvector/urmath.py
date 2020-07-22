import urvector
import numpy as np

def rot_scale(rx, ry, rz):
    # rotation data from sources such as Secondary Interface doesn't match PolyScope.
    # The values are viable and scaling is not required but if desired rot_scale() scales the values to match PolyScope.
    if ((abs(rx) >= 0.001 and rx < 0.0) or (abs(rx) < 0.001 and abs(ry) >= 0.001 and ry < 0.0) or \
            (abs(rx) < 0.001 and abs(ry) < 0.001 and rz < 0.0)):
        scale = 1 - 2 * pi / length([rx, ry, rz])
        output = [scale * rx, scale * ry, scale * rz]
    else:
        output = [rx, ry, rz]
    return output


def pose_trans(pose_from, pose_to):
    """
    Duplicates functionality of URScript pose_trans()
    :param pose_from: reference frame (type: Pose)
    :param pose_to:  offset frame (type: Pose)
    :return: Pose object
    """
    transmat_from = urvector.TransformationMatrix.fromPose(pose_from).toArray()
    transmat_to = urvector.TransformationMatrix.fromPose(pose_to).toArray()
    transmat_result = urvector.TransformationMatrix.fromArray(np.matmul(transmat_from, transmat_to))
    return urvector.Pose(transmat_result.position, transmat_result.rotation)

def pose_add(pose_from, pose_to):
    """
    Duplicates functionality of URScript pose_trans()
    :param pose_from: reference frame (type: Pose)
    :param pose_to:  offset frame (type: Pose)
    :return: Pose object
    """
    position_from, _ = pose_from.split()
    position_to, _ = pose_to.split()
    output_position =   urvector.Position(position_from.x + position_to.x, position_from.y + position_to.y,
                                          position_from.z + position_to.z)

    transmat_from = urvector.TransformationMatrix.fromPose(pose_from).toArray()
    transmat_to = urvector.TransformationMatrix.fromPose(pose_to).toArray()
    transmat_result = urvector.TransformationMatrix.fromArray(np.matmul(transmat_from, transmat_to))
    return urvector.Pose(output_position, transmat_result.rotation)

