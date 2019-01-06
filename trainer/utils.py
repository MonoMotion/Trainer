from roboschool.scene_abstract import cpp_household

import flom

def dictzip(d1, d2):
    for k, v in d1.items():
        yield k, (v, d2[k])

def select_location(ty, vec, root_pose):
    if ty == flom.CoordinateSystem.World:
        return vec
    elif ty == flom.CoordinateSystem.Local:
        cpose = cpp_household.Pose()
        cpose.set_xyz(*vec)
        return root_pose.dot(cpose).xyz()
    else:
        assert False  # unreachable

def select_rotation(ty, quat, root_pose):
    if ty == flom.CoordinateSystem.World:
        return quat
    elif ty == flom.CoordinateSystem.Local:
        cpose = cpp_household.Pose()
        cpose.set_quaternion(*quat)
        return root_pose.dot(cpose).quatertion()
    else:
        assert False  # unreachable
