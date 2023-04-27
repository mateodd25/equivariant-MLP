#!/usr/bin/env python3

from emlp.utils import export
from .groups import S, O


class GroupSequence(object):
    """Abstract sequence of groups."""

    def group(self, j):
        """Return group at level j."""
        return NotImplementedError


@export
class PermutationGroupSequence(GroupSequence):
    """Sequence of permutation groups."""

    def __init__(self):
        """Initiate lazy sequence."""
        self.name = "permutation"

    def group(self, j):
        """Return the group at level j."""
        return S(j)


@export
class OrthogonalGroupSequence(GroupSequence):
    """
    Sequence of orthogonal groups O(n).
    """

    def __init__(self):
        """Initialize laxy sequnece."""
        self.name = "orthogonal"

    def group(self, j):
        return O(j)
