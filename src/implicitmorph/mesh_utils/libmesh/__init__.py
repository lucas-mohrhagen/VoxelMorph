'''
From Occupancy Networks repository: https://github.com/autonomousvision/occupancy_networks
'''

from .inside_mesh import (
    check_mesh_contains, MeshIntersector, TriangleIntersector2d
)


__all__ = [
    check_mesh_contains, MeshIntersector, TriangleIntersector2d
]

# source: Max
# to install
# cd into libmesh
# mkdir libmesh
# python setup.py build_ext --inplace
# move file from libmesh/libmesh to libmesh/
# voila