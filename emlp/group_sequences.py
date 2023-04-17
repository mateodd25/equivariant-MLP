#!/usr/bin/env python3

from emlp.utils import export
from .groups import S


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
