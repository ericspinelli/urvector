"""
Vector classes for robotics pose data for Universal Robots

Defines generic Vector class which is inherited by:
Pose            p[x,y,z,rx,ry,rz]
Position        [x,y,z]
Rotation Vector [rx,ry,rz]
Joint           [j0,j1,j2,j3,j4,j5]

Defines generic Matrix which inherits Vector and is inherited by:
Rotation Matrix         [3x3]
Transformation Matrix   [4x4]
"""

__version__ = 0.1
__author__ = "Eric Spinelli"

import numpy as np
import math

class VectorError(Exception):
    pass


class Vector(object):
    """
    Vector class represents multi-dimension vectors where order of components is important
    It is intended as a parent class for classes with defined dimensions and order

    :param __dimensions: number of dimensions
    :type __dimensions: int
    :param __dimensions_list: list of dimension labels
    :type __dimensions_list: list
    :param __order: order for each dimension
    :type __order: dict
    """

    def __init__(self, order=None, **kwargs):
        """
        Constructor
        :param order: defines order of dimensions (defined in each sub-class)
        :param kwargs: dimensions labels (str) and values (float)
        """
        self._dimensions = 0
        self._dimensions_list = []
        if order:
            self._order = order
        else:
            self._order = {}

        for key, value in kwargs.items():
            if order is None:
                self._order[key] = self._dimensions
            self._dimensions += 1
            self._dimensions_list.append(key)
            setattr(self, key, value)

    @classmethod
    def fromList(cls, input_list):
        """
        Alterative constructor from a list.  Requires size parameter to be defined by sub-class.
        :param input_list: dimension values (dimension labels are determined by sub-class)
        :return: instance of sub-class
        """
        if len(input_list) > cls.size:
            raise IndexError
        return cls(*input_list)

    def __repr__(self):
        return str(self.toList())

    def __len__(self):
        return self._dimensions

    def isValid(self, size=None):
        if size and self._dimensions != size:
            return False
        for key in self._dimensions_list:
            value = getattr(self, key)
            if value is None:
                return False
            elif not isinstance(value, (int, float)):
                return False
            elif isinstance(value, bool):
                return False
        return True

    def getDimensions(self):
        return self._dimensions_list

    def toList(self):
        output = []
        for dimension in sorted(self._dimensions_list, key=lambda i: self._order[i]):
            output.append(getattr(self, dimension))
        return output

    def toArray(self):
        return np.array(self.toList())


class Pose(Vector):
    """
    :param size: 6-dimension vector
    :param order: order of dimensions (x, y, z, rx, ry, rz) for Python versions earlier than v3.6
    """
    size = 6
    order = {'x': 0, 'y': 1, 'z': 2, 'rx': 3, 'ry': 4, 'rz': 5} 

    def __init__(self, x=None, y=None, z=None, rx=None, ry=None, rz=None):
        """
        Constructor: can take Position (x, y, z) and Rotation (rx, ry, rz) or 6 individual values
        :param x: Position object (x, y, z) OR x value
        :param y: Rotation object (rx, ry, rz) OR y value
        :param z: z value
        :param rx: rx value
        :param ry: ry value
        :param rz: rz value
        """
        if isinstance(x, Position) and isinstance(y, Rotation):
            Vector.__init__(self, order=self.order, x=x.x, y=x.y, z=x.z, rx=y.rx, ry=y.ry, rz=y.rz)
        else:
            Vector.__init__(self, order=self.order, x=x, y=y, z=z, rx=rx, ry=ry, rz=rz)

    def __repr__(self):
        """
        :return: output Pose in form p[x, y, z, rx, ry, rz] to match URScript
        """
        return "p" + Vector.__repr__(self)

    def split(self):
        """
        :return: returns two objects of type Position [x, y, z] and Rotation [rx, ry, rz]
        """
        if self.isValid():
            pos = Position(self.x, self.y, self.z)
            rot = Rotation(self.rx, self.ry, self.rz)
            return pos, rot


class Position(Vector):
    """
    :param size: 3-dimension vector
    :param order: order of dimensions (x, y, z) for Python versions earlier than v3.6
    """
    size = 3
    order = {'x': 0, 'y': 1, 'z': 2}

    def __init__(self, x=None, y=None, z=None):
        Vector.__init__(self, order=self.order, x=x, y=y, z=z)


class Rotation(Vector):
    """
    :param size: 3-dimension vector
    :param order: order of dimensions (rx, ry, rz) for Python versions earlier than v3.6
    """
    size = 3
    order = {'rx': 0, 'ry': 1, 'rz': 2}

    def __init__(self, rx=None, ry=None, rz=None):
        Vector.__init__(self, order=self.order, rx=rx, ry=ry, rz=rz)


class Joint(Vector):
    """
    :param size: 6-dimension vector
    :param order: order of dimensions (j0, j1, j2, j3, j4, j5) for Python versions earlier than v3.6
    """
    size = 6
    order = {'j0': 0, 'j1': 1, 'j2': 2, 'j3': 3, 'j4': 4, 'j5': 5}

    def __init__(self, j0=None, j1=None, j2=None, j3=None, j4=None, j5=None):
        Vector.__init__(self, order=self.order, j0=j0, j1=j1, j2=j2, j3=j3, j4=j4, j5=j5)


class MatrixError(Exception):
    # def __init__(self, msg):
    #     self.message = msg
    pass


class Matrix(Vector):
    """
    Matrix objects are square Vectors (rows & columns) of X dimensions
    """
    def toList(self):
        """
        Overloaded Vector.toList() to handle X-dimensions split into rows & columns
        :return: list of lists (outer list: size "rows", inner list: size "columns")
        """
        output = [[] for i in range(self.size2D[0])]
        sort = sorted(self._dimensions_list, key=lambda i: self._order[i])
        j = 0
        for i in range(len(output)):
            for _ in range(self.size2D[1]):
                output[i].append(getattr(self, sort[j]))
                j += 1
        return output


class RotationMatrix(Matrix):
    """
    :param size: 9-dimension vector (3x3)
    :param order: order of dimensions (r11, r12, r13, r21, r22, r23, r31, r32, r33) for Python versions earlier than v3.6
    """
    size2D = (3, 3)
    size = size2D[0] * size2D[1]
    order = {'r11': 0, 'r12': 1, 'r13': 2,
             'r21': 3, 'r22': 4, 'r23': 5,
             'r31': 6, 'r32': 7, 'r33': 8}

    def __init__(self, r11, r12, r13, r21, r22, r23, r31, r32, r33):
        Vector.__init__(self, order=self.order, r11=r11, r12=r12, r13=r13, r21=r21, r22=r22, r23=r23, r31=r31, r32=r32, r33=r33)

    @classmethod
    def fromRotation(cls, rotation):
        """
        Converts a Rotation into a RotationMatrix
        :param rotation: Rotation (3D Vector) object
        :return: RotationMatrix (9D Vector) object
        """
        rx, ry, rz = rotation.toList()
        theta = math.sqrt(math.pow(rx, 2) + math.pow(ry, 2) + math.pow(rz, 2))
        if theta != 0:
            ux = rx / theta
            uy = ry / theta
            uz = rz / theta
            cth = math.cos(theta)
            sth = math.sin(theta)
            vth = 1 - math.cos(theta)
        else:
            ux, uy, uz, sth, vth = 0, 0, 0, 0, 0
            cth = 1

        # column 1
        r11 = ux * ux * vth + cth
        r21 = ux * uy * vth + uz * sth
        r31 = ux * uz * vth - uy * sth
        # column 2
        r12 = ux * uy * vth - uz * sth
        r22 = uy * uy * vth + cth
        r32 = uy * uz * vth + ux * sth
        # column 3
        r13 = ux * uz * vth + uy * sth
        r23 = uy * uz * vth - ux * sth
        r33 = uz * uz * vth + cth

        return cls(r11, r12, r13, r21, r22, r23, r31, r32, r33)

    @classmethod
    def fromPose(cls, pose):
        """
        Coverts Pose object into RotationMatrix object
        :param pose: Pose object (6D Vector)
        :return: RotationMatrix object (9D Vector)
        """
        _ , rotation = pose.split()
        return cls.fromRotation(rotation)

    def toRotation(self):
        """
        :return: Rotation object (3D Vector)
        """
        theta = math.acos((self.r11 + self.r22 + self.r33 - 1) / 2)
        sth = math.sin(theta)

        if (theta > math.radians(179.99)) or (theta < math.radians(-179.99)):
            theta = math.radians(180)
            if (self.r21 < 0):
                if (self.r31 < 0):
                    ux = math.sqrt((self.r11 + 1) / 2)
                    uy = -math.sqrt((self.r22 + 1) / 2)
                    uz = -math.sqrt((self.r33 + 1) / 2)
                else:
                    ux = math.sqrt((self.r11 + 1) / 2)
                    uy = -math.sqrt((self.r22 + 1) / 2)
                    uz = math.sqrt((self.r33 + 1) / 2)
            else:
                if (self.r31 < 0):
                    ux = math.sqrt((self.r11 + 1) / 2)
                    uy = math.sqrt((self.r22 + 1) / 2)
                    uz = -math.sqrt((self.r33 + 1) / 2)
                else:
                    ux = math.sqrt((self.r11 + 1) / 2)
                    uy = math.sqrt((self.r22 + 1) / 2)
                    uz = math.sqrt((self.r33 + 1) / 2)
        else:
            if sth != 0:
                ux = (self.r32 - self.r23) / (2 * sth)
                uy = (self.r13 - self.r31) / (2 * sth)
                uz = (self.r21 - self.r12) / (2 * sth)
            else:
                ux = 0
                uy = 0
                uz = 0

        return Rotation((theta * ux), (theta * uy), (theta * uz))


class TransformationMatrix(Matrix):
    """
    :param size: 16-dimension vector (4x4) that contains a Position and RotationMatrix
    :param order: order of dimensions (r11, r12, r13, p1, r21, r22, r23, p2, r31, r32, r33, p2, z1=0, z2=0, z3=0, z4=1)
    :param position: @property outputing a Position object
    :param rotation: @property outputing a Rotation object
    :param rotmat: @property outputing a RotationMatrix object
    """
    size2D = (4, 4)
    size = size2D[0] * size2D[1]
    order = {'r11': 0, 'r12': 1, 'r13': 2, 'p1': 3,
             'r21': 4, 'r22': 5, 'r23': 6, 'p2': 7,
             'r31': 8, 'r32': 9, 'r33': 10, 'p3': 11,
             'z1': 12, 'z2': 13, 'z3': 14, 'z4': 15}

    @classmethod
    def fromPose(cls, pose):
        """
        :param pose: Pose object (6D Vector)
        :return: TransformationMatrix object (16D Vector)
        """
        position, rotation = Pose.split(pose)
        rotmat = RotationMatrix.fromRotation(rotation)
        return cls(order=cls.order,
                   r11=rotmat.r11, r12=rotmat.r12, r13=rotmat.r13, p1=position.x,
                   r21=rotmat.r21, r22=rotmat.r22, r23=rotmat.r23, p2=position.y,
                   r31=rotmat.r31, r32=rotmat.r32, r33=rotmat.r33, p3=position.z,
                   z1=0, z2=0, z3=0, z4=1)

    @classmethod
    def fromArray(cls, array):
        """
        Creates a TransformationMatrix from a list of lists, such as the return value of np.array()
        :param array: numpy style array [[], [], []] etc.
        :return: TransformationMatrix object
        """
        return cls(r11=array[0][0], r12=array[0][1], r13=array[0][2], p1=array[0][3],
                   r21=array[1][0], r22=array[1][1], r23=array[1][2], p2=array[1][3],
                   r31=array[2][0], r32=array[2][1], r33=array[2][2], p3=array[2][3],
                   z1=0, z2=0, z3=0, z4=1)

    @property
    def position(self):
        return Position(self.p1, self.p2, self.p3)

    @property
    def rotation(self):
        rotmat = self.rotmat
        return rotmat.toRotation()

    @property
    def rotmat(self):
        return RotationMatrix(self.r11, self.r12, self.r13,
                              self.r21, self.r22, self.r23,
                              self.r31, self.r32, self.r33)


def test_vector():
    from urmath import pose_trans, pose_add

    d6_list = [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, None, 5, 6],
        ['a', 2, 3, 4, 5, 6],
        [1, True, 3, 4, 5, 6],
        [6, 5, 4, 3, 2, 1]
    ]

    for data in d6_list:
        p = Pose.fromList(data)
        print("Pose: {} is Valid: {}".format(p, p.isValid()))
    print("")

    p_from = Pose(0.2, 0.1, 0.2, 0, 3.14, 0)
    p_to = Pose(0.1, 0.04, 0, 0, 0, 0.785)
    ptrans_result = pose_trans(p_from, p_to)
    print("pose_trans({}, {}) = {}".format(p_from, p_to, ptrans_result))
    print("")

    p_from = Pose(0.05, 0.1, 0.24, 0.22, 0.22, 0)
    p_to = Pose(0.1, 0.1, 0, 0, 0, 0.785)
    padd_result = pose_add(p_from, p_to)
    ptrans_result = pose_trans(p_from, p_to)
    print("pose_add({}, {}) = {}".format(p_from, p_to, padd_result))
    print("pose_trans({}, {}) = {}".format(p_from, p_to, ptrans_result))
    print("")

    p, r = p_from.split()
    print("Pose.split() = {}, {}".format(p, r))

    for data in d6_list:
        j = Joint.fromList(data)
        print("Joint: {} is Valid: {}".format(j, j.isValid()))
    print("")

    d3_list = [
        [1, 2, 3],
        [3, 2, 1],
        [1, 2, None],
        ['a', 2, 3],
        [1, True, 3]
    ]

    for data in d3_list:
        p = Position.fromList(data)
        print("Position: {} is Valid: {}".format(p, p.isValid()))
    print("")

    for data in d3_list:
        r = Rotation.fromList(data)
        print("Rotation: {} is Valid: {}".format(r, r.isValid()))
    print("")

    p = Pose(1, 2, 3, 4, 5, 6)
    print("Pose: {} is Valid: {}".format(p, p.isValid()))
    p = Position(1, 2, 3)
    print("Position: {} is Valid: {}".format(p, p.isValid()))
    r = Rotation(4, 5, 6)
    print("Rotation: {} is Valid: {}".format(r, r.isValid()))
    p1 = Pose(p, r)
    print("Pose: {} is Valid: {}".format(p1, p1.isValid()))
    print("p: ", p.x, p.y, p.z)
    print("r: ", r.rx, r.ry, r.rz)
    print("p1: ", p1.x, p1.y, p1.z, p1.rx, p1.ry, p1.rz)
    print("j: ", j.j0, j.j1, j.j2, j.j3, j.j4, j.j5)
    print("")

def test_matrix():
    p = Pose.fromList([1, 2, 3, 0, 3.14, 0])
    rm = RotationMatrix.fromPose(p)
    tm = TransformationMatrix.fromPose(p)
    print("p: {}".format(p))
    print("rm: {}".format(rm))
    print("r12: {}, r33: {}".format(rm.r12, rm.r33))
    print("tm: {}".format(tm))
    print("")

    r = rm.toRotation()
    print("r: {}".format(r))
    print("tm.position: {}".format(tm.position))
    print("tm.rotation: {}".format(tm.rotation))
    print("tm.rotmat: {}".format(tm.rotmat))
    print("")

    d6_list = [
        [1, 2, 3, 4, 5, 6],
        [6, 5, 4, 3, 2, 1]
    ]

    print("from list")
    for data in d6_list:
        p = Pose.fromList(data)
        tm = TransformationMatrix.fromPose(p)
        print("p: {}".format(p))
        print("p.isValid(): {}".format(p.isValid()))
        print("tm.isValid(): {}".format(tm.isValid()))
        print("tm.position.isValid(): {}".format(tm.position.isValid()))
        print("tm.rotation.isValid(): {}".format(tm.rotation.isValid()))
        print("tm.rotmat.isValid(): {}".format(tm.rotmat.isValid()))
    print("")

if __name__ == "__main__":
    test_vector()
    test_matrix()