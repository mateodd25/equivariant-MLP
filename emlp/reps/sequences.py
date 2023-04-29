#!/usr/bin/env python3

#!/usr/bin/env python3
import numpy as np
from emlp.utils import export
from plum import dispatch
from ..group_sequences import PermutationGroupSequence, OrthogonalGroupSequence
from .linear_operators import I, LazyDirectSum, LazyKron, SlicedI, lazify, ConcatLazy
from .representation import ScalarRep, V
from .utils import null_space
from functools import reduce
from collections import defaultdict

@export
class ConsistentSequence(object):
    r"""Consistent sequence of representations."""

    presentation_degree = NotImplemented
    generation_degree = NotImplemented

    def group(self, j):
        """Group at level j."""
        return self.group_sequence().group(j)

    def group_sequence(self):
        """Group sequence"""
        raise NotImplementedError

    def representation(self, j):
        """Representation at level j."""
        return V(self.group(j))

    def dimension(self, j):
        """Representation dimension at level j."""
        return self.representation(j).size()

    def up_embedding(self, j):
        r"""Embedding going from the representation at level j to level j+1."""
        raise NotImplementedError

    def composite_embedding(self, n, i):
        """Composite embedding from level n to i.

        That is, embedding(n-1) @ ... @ embedding(i).
        """

        if n <= i:
            raise ValueError(
                f"First input {n} has to be greather than the second input {i}."
            )
        elif n == i + 1:
            return self.up_embedding(i)
        else:  # n > j + 1
            return self.up_embedding(n - 1) @ self.composite_embedding(n - 1, i)

    def extendability_constraints(self, n, n0):
        """Gives constraints that extend an element at level n0 to level n."""

        constraints = []
        constraints.append(self.representation(n).constraint_matrix())
        constraints.append(self.composite_embedding(n, n0).H)
        return ConcatLazy(constraints)

    def __add__(self, other):
        """Direct sum of two representations"""
        if isinstance(other, int):
            if other == 0:
                return self
            else:
                return self + other * TrivialSequence(
                    group_sequence=self.group_sequence()
                )
        return SumSequence(self, other)

    def __mul__(self, other):
        return mul_sequences(self, other)

    def __rmul__(self, other):
        return mul_sequences(other, self)

@export 
def gated(sequence: ConsistentSequence) -> ConsistentSequence:
    """
        
    """

    if 

@export 
class GatedSequence(ConsistentSequence): 
    def __init__():
        

    
# --------------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------------
# The  following define sums and tensor products of consistent sequences.
# *Warning* The way we compute the presentation and generation degree only works
# for permutation groups. It is unclear how to compute these for general representations.
# TODO: Have a more robust way to compute these degrees for other representations.


class SumSequence(ConsistentSequence):
    """Sum sequence between two sequences"""

    def group_sequence(self):
        """Group sequence shared by both summands"""
        return self._group_sequence

    # def __init__(self, *sequences):

    def __init__(self, *sequences):
        sequences = [
            SumSequenceFromCollection({TrivialSequence: seq}) if isinstance(seq, int) else seq 
            for seq in sequences
        ]
        seq_counters = [
            seq.sequences if isinstance(seq, SumSequence) else {seq: 1} for seq in sequences 
        ]
        self.sequences = self._compute_canonical(seq_counters)
        self.generation_degree = reduce(max, [seq.generation_degree for seq in self.sequences.keys()])       
        self.presentation_degree = reduce(max, [seq.presentation_degree for seq in self.sequences.keys()])       
        self._group_sequence = next(iter(self.sequences)).group_sequence()
        self.is_permutation = all([seq.is_permutation for seq in self.sequences.keys()])
        # self._num_summands = sum([ for ])
        
    def _compute_canonical(self, seq_counters):
        unique_seq = sorted(reduce(lambda a, b: a | b, [seq.keys() for seq in seq_counters]))
        merged_counts = defaultdict(int)
        ids = [0] * len(seq_counters)
        for pair in seq_counters:
            seq = next(iter(pair.keys()))
            counter = pair[seq]
            merged_counts[seq] += counter 
        return merged_counts
            
    def num_sumands(self):
        return len(sel)
        
    def __init__(self, first_sequence, second_sequence):
        """Initialize with the two summands."""
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence
        self.presentation_degree = max(
            first_sequence.presentation_degree, second_sequence.presentation_degree
        )
        self.generation_degree = max(
            first_sequence.generation_degree, second_sequence.generation_degree
        )
        self.is_permutation = first_sequence.is_permutation and second_sequence.is_permutation

    def representation(self, j):
        """Direct sum representation"""
        rep = self.first_sequence.representation(
            j
        ) + self.second_sequence.representation(j)
        rep.G = self.group_sequence().group(j)
        return rep

    def up_embedding(self, j):
        """Direct sum of the embeddings"""
        return LazyDirectSum(
            [self.first_sequence.up_embedding(j), self.second_sequence.up_embedding(j)]
        )

class SumSequenceFromCollection(SumSequence):
    def __init__(self, counter, perm=None):
        self.sequences = counter
        self.perm = np.arange(self.num_sumands()) if perm is None else perm
        self.sequences, self.perm = self._compute_canonical([counter], [self.perm])
        self.invperm = np.argsort(self.perm)
        self.is_permutation = all(rep.is_permutation for rep in self.sequences.keys())
    
class ProductSequence(ConsistentSequence):
    """Product sequence between two sequences."""

    def group_sequence(self):
        return self.first_sequence.group_sequence()

    def __init__(self, first_sequence, second_sequence):
        self.first_sequence = first_sequence
        self.second_sequence = second_sequence
        self.presentation_degree = max(
            first_sequence.presentation_degree + second_sequence.generation_degree,
            first_sequence.generation_degree + second_sequence.presentation_degree,
        )
        self.generation_degree = (
            first_sequence.generation_degree + second_sequence.generation_degree
        )
        self.is_permutation = first_sequence.is_permutation and second_sequence.is_permutation

    def representation(self, j):
        r"""Tensor product representation"""
        return self.first_sequence.representation(
            j
        ) * self.second_sequence.representation(j)

    def up_embedding(self, j):
        r"""Kroneker product of the embeddings"""
        return LazyKron(
            # [self.first_sequence.up_embedding(j), self.second_sequence.up_embedding(j)]
            [self.second_sequence.up_embedding(j), self.first_sequence.up_embedding(j)]
        )


@dispatch
def mul_sequences(sequence_a, sequence_b: int):
    if sequence_b == 1:
        return sequence_a
    elif sequence_b == 0:
        return 0
    else:
        return SumSequence(*(sequence_b * [sequence_a]))


@dispatch
def mul_sequences(sequence_a: int, sequence_b):
    return mul_sequences(sequence_b, sequence_a)


@dispatch
def mul_sequences(sequence_a, sequence_b):
    if type(sequence_a) is TrivialSequence:
        return sequence_b
    if type(sequence_b) is TrivialSequence:
        return sequence_a
    return ProductSequence(sequence_a, sequence_b)


# --------------------------------------------------------------------------------
# Linear map sequences
# --------------------------------------------------------------------------------
# We handle operators sequences in a different fashion than general representations
# in order to have direct access to the embeddings of both input and output
# representations.


@export
class EquivariantOperatorSequence(object):
    """Sequence of equivariant linear operator spaces."""

    def __init__(self, input_representation, output_representation=None):
        """Init method."""
        self.input_representation = input_representation
        if output_representation is None:
            self.output_representation = input_representation
        else:
            self.output_representation = output_representation

            # self.presentation_degree = max(
        #     self.input_representation.presentation_degree,
        #     self.output_representation.presentation_degree,
        # )

    def compatibility_constraints(self, j):
        """Constraints that ensure that the basis at level j is extendable."""
        constraints = []
        pre_dgr = self.input_representation.presentation_degree
        if j < pre_dgr:
            raise ValueError(
                f"Can only extend when the level {j} is equal or larger than the presentation degree {pre_dgr}"
            )

        constraints.extend(
            [
                # Their kronecker version is backwards why?!
                LazyKron(
                    [
                        (
                            I(self.output_representation.dimension(j))
                            - self.output_representation.composite_embedding(j, k)
                            @ self.output_representation.composite_embedding(j, k).H
                        ),
                        self.input_representation.composite_embedding(j, k).H,
                    ]
                )
                for k in range(
                    1, min(self.input_representation.presentation_degree, j - 1) + 1
                )
            ]
        )
        return ConcatLazy(constraints)

    def equivariant_basis(self, level):
        return self.at_level(level).equivariant_basis()

    def composite_embedding(self, up_level, low_level):
        return LazyKron(
            [
                self.output_representation.composite_embedding(up_level, low_level),
                self.input_representation.composite_embedding(up_level, low_level),
            ]
        )

    def extendability_constraints(self, n, n0):
        constraints = []
        constraints.append(
            (
                self.input_representation.representation(n)
                >> self.output_representation.representation(n)
            ).constraint_matrix()
        )
        constraints.append(
            LazyKron(
                [
                    self.output_representation.composite_embedding(n, n0).H,
                    self.input_representation.composite_embedding(n, n0).H,
                ]
            )
        )
        return ConcatLazy(constraints)

    def at_level(self, j):
        return EquivariantOperators(
            self.input_representation.representation(j),
            self.output_representation.representation(j),
            self.compatibility_constraints(j),
        )


@export
class EquivariantOperators(object):
    def __init__(
        self,
        input_representation,
        output_representation,
        compatibility_constraints=None,
    ):
        self.input_representation = input_representation
        self.output_representation = output_representation
        self.compatibility_constraints = compatibility_constraints

    def equivariant_basis(self):
        linear_maps = self.input_representation >> self.output_representation
        basis = linear_maps.equivariant_basis()
        if self.compatibility_constraints is not None:
            coefficients = null_space(self.compatibility_constraints @ lazify(basis))
            basis = basis @ coefficients
        return basis


# --------------------------------------------------------------------------------
# Implementations of consistent sequences
# --------------------------------------------------------------------------------


@export
class TrivialSequence(ConsistentSequence):
    r"""Trivial sequence, the representations at all levels are 1."""

    def __init__(self, group_sequence=None):
        self._group_sequence = group_sequence
        self.presentation_degree = 1
        self.generation_degree = 1
        self.is_permutation = True

    def group(self, j):
        """Return the group at level j."""
        if self.group_sequence is None:
            return None
        return self.group_sequence.group(j)

    def group_sequence(self):
        """Return group sequence."""
        return self._group_sequence

    def representation(self, j):
        """Return trivial representation."""
        return ScalarRep(self.group_sequence().group(j))

    def up_embedding(self, j):
        """Identity embedding."""
        return lazify(np.eye(1))


@export
class PermutationSequence(ConsistentSequence):
    """Permutation group representation sequence."""

    def __init__(self):
        """Initialize the sequence."""
        self.presentation_degree = 2
        self.generation_degree = 2
        self._group_sequence = PermutationGroupSequence()
        self.is_permutation = True

    # def group(self, j):
    # return self.group_sequence().group(j)

    def group_sequence(self):
        """Return the permutation group in j elements."""
        return self._group_sequence

    def up_embedding(self, j):
        """Pad with one zero."""
        return SlicedI(j + 1, j)


@export
class OrthogonalSequence(ConsistentSequence):
    """Orthogonal group representation sequence."""

    def __init__(self):
        """Initialize the sequence."""
        self.presentation_degree = (
            1  # It is unclear whether this is the case, but it seems to be true.
        )
        self.generation_degree = 1
        self._group_sequence = OrthogonalGroupSequence()
        self.is_permutation = False

    # def group(self, j):
    # return self.group_sequence().group(j)

    def group_sequence(self):
        """Return the permutation group in j elements."""
        return self._group_sequence

    def up_embedding(self, j):
        """Pad with one zero."""
        return SlicedI(j + 1, j)
