======
Theory
======

This section will contain the theoretical background of the code. It will be
divided into several subsections, each of which will contain a brief
introduction to the topic, followed by a more detailed explanation of the
relevant concepts.

Introduction
------------

PgOP's main use case is to determine if a point in space (which can belong to a particle
location, or not) has a bond orientation order diagram symmetry of a given point group.
It is important to point out that PgOP does NOT calculate the Wyckoff site symmetry or
crystalline point group symmetry for a particle in crystalline environment (unless this
particle sits at the general position).
Symmetry is defined as a binary relation between two objects that are the same under
some transformation. Some objects or points in space or systems can either have some
symmetry or not. PgOP removes this binary distinction and instead gives a continuous
measure of how symmetric the object or set of points is with respect to some symmetry.
Such approach can be incredibly useful when studying for example the local structure of
a crystal as it is formed.

Symmetry is a fundamental concept in physics and mathematics. It is a property of an
object that remains unchanged under some transformation. For example, a square has
rotational symmetry of order 4, because it looks the same after a 90 degree rotation.
Symmetry can be described in terms of a group of transformations that leave the object
invariant. This group is called the symmetry group of the object. There are several
types of symmetry operations that are often encountered in physics, such as rotations,
reflections, and translations. These operations can be combined to form more complex 
symmetry operations.

There are two main types of symmetry operations associated with point groups: rotations
and reflections. Inversion can be considered as a special case of reflection, while
rotoreflections can be constructed using combination of rotations and inversions. In
space groups in addition to the above-mentioned symmetry elements we also have
translational symmetry. This symmetry can be combined with elements of point group
symmetry to obtain new unique type of symmetry operations such as screw axes (rotation +
translation) and glide planes (reflections + translations). Since PgOP computes only
point group symmetry we shall focus only on point group symmetry.

Symmetry operations in point groups
-----------------------------------

Rotations are defined by the axis of rotation and the angle of rotation.  Various
representations of rotations exist, each with distinct advantages and disadvantages. In
PgOP, we primarily use the Euler angle representation in the zyz convention because the
relevant literature often adopts it :cite:`Altmann_WignerD`. A rotation operation is
written as :math:`\hat{C}_{nz}`, where :math:`n` represents the order of rotation and
the second letter indicates the rotation axis. Other axes besides :math:`x`, :math:`y`,
or :math:`z` can be used. If the operation is written without an explicit rotation axis,
such as :math:`\hat{C}_n`, it denotes the main rotation axis (aligned with the rotation
axis of the highest rotation order), typically taken to be the :math:`z` axis. The angle
of rotation (:math:`\theta`) can be computed from the order :math:`n` using the formula:
:math:`\theta = 2\pi/n = 360^\circ / n`. Multiple consecutive rotations are often
applied in group theory and are written using power notation: :math:`\hat{C}_n^m`. This
notation means that the operation :math:`\hat{C}_n` is applied :math:`m` times in
succession.

Reflections are defined by the plane of reflection. Reflections are a type of symmetry
operation that flips the object across the plane of reflection. Reflections cannot be
represented as rotations, or combination of rotations in a general sense. Reflections
can be represented as inversion followed by a rotation of 180 degrees
:cite:`Altmann_WignerD`:

.. math::
    \hat{\sigma}_{xy} = \hat{i} \hat{C}_{2z} \\
    \hat{\sigma}_h = \hat{i} \hat{C}_2

where :math:`\hat{i}` is the inversion operator and :math:`\hat{C}_2(z)` is the two fold
rotation around :math:`z` axis. The reflection plane is always perpendicular to the axis
of rotation obtained by the above formula.

Inversion is a symmetry operation that flips the object across the center of inversion.
It can be shown that inversion can be represented as application of 3 orthogonal
reflection :cite:`engel2021point`:

.. math::
    \hat{i} = \hat{\sigma}_{yz} \hat{\sigma}_{xz} \hat{\sigma}_{xy}

Rotoreflections are a combination of rotations and reflections, sometimes called
improper rotations. They are a type of symmetry operation that combines rotation and
reflection. Thus, by definition, we can write :cite:`Altmann_WignerD`: 

.. math::
    \hat{S}_n = \hat{\sigma}_h {\hat{C}_n} = \hat{\sigma}_{xy} {\hat{C}_n}

where :math:`\hat{\sigma}_h=\hat{\sigma}_{xy}` is the reflection operator perpendicular
to the axis of rotation (:math:`z`).

Some useful equivalency relations for rotoreflections and their powers used in PgOP code can be found in work by Drago :cite:`drago1992`.

Group theory
------------

In group theory, sets with operation under certain constraints (operation must be
associative, and have an identity element, and every element of the set must have an inverse) are
called groups. When studying symmetry groups, we usually consider groups under operation
of composition. The elements of the group are symmetry operations. Elements of the group
can act on many different objects such as Euclidian space, or physical or other
geometrical objects built from such an object (for example shapes or points). Euclidian
(or other types of spaces) can often be described as vector spaces.

Another important aspect of the group is the group action. First, let's consider a
general action of some element of group :math:`G`. Let :math:`G` be a group under
composition. Consider an action of an element of group :math:`G`, say operator :math:`g`
on some function :math:`f`. The action of :math:`g` on :math:`f` is just the composition
of :math:`g` on :math:`f`. If we assume that :math:`G` is a symmetry group, then the
interpretation of this composition is that action of :math:`g` symmetrizes the function
:math:`f` according to symmetry operator :math:`g`. Similarly, we can also apply a group
action of the group :math:`G` onto some function :math:`f`. The group action is
symmetrization under all the elements (symmetry operators) of the group. If we assume
that :math:`G` is a finite point group, the group action is given by the following
formula:

.. math::
    f_G = \frac{1}{|G|} \sum_{g \in G} g \cdot f,

where :math:`|G|` is the order of the group (number of elements of :math:`G`).

When group action acts on a vector space we call this a representation. Notice that
choosing a representation enables us to actually numerically write out the operator in a
matrix form.

Wigner D matrices
~~~~~~~~~~~~~~~~~
Symmetry operations can be represented as matrices acting on a vector space. These will
be different based on the representation we chose. One such choice is the Wigner D
matrix, which are matrices representing symmetry operations in the space spanned by
spherical harmonics. Spherical harmonics are a set of functions which make a complete
basis in the space of functions on the sphere. This is exactly what we will need for
PgOP and choice for it will become apparent later. For now, we focus our attention on
how to construct these matrices for basic operations.

A single Wigner :math:`D` matrix is defined for a given symmetry operation and a given :math:`l`, which
is the degree of the spherical harmonic. The Wigner :math:`D` matrix is a square matrix of size
:math:`2l+1`. The indices of the matrix are often written as :math:`m` and :math:`m'`
and they range from :math:`-l` to :math:`l`. The vectors which these matrices operate on
are coefficients for a spherical harmonic given by :math:`l` and :math:`m` (each vector
element is different :math:`m`).

A single Wigner :math:`D` matrix is defined for a given symmetry operation and a given
:math:`l`, which is the degree of the spherical harmonic. The Wigner :math:`D` matrix is a
square matrix of size :math:`2l+1`. The indices of the matrix are often written as :math:`m` and
:math:`m'` and they range from :math:`-l` to :math:`l`. The vectors which these matrices operate on
are coefficients for a spherical harmonic given by :math:`l` and :math:`m` (each vector element
is different :math:`m`).

First, we give the formula for composition operation which is just a matrix
multiplication. Matrix multiplication (composition) formula for two symmetry operations
is given by:

.. math::
    D^{(l)}_{m'm''}(g_1) \times D^{(l)}_{m''m}(g_2) = D^{(l)}_{m'm}(g_1 g_2) = \sum_{m''=-l}^l D^{(l)}_{m'm''}(g_1) D^{(l)}_{m''m}(g_2)


In PgOP code we use Wigner :math:`D` matrices to represent symmetry operations. We use
the zyz convention to generate the matrices and follow the work of Altmann
:cite:`Altmann_WignerD`. 


Group action of Wigner D matrices
*********************************

Group action formula can be given in terms of Wigner D matrices. The group action is a
matrix which can be constructed by summing Wigner D matrices of operations in a group:

.. math::
    D^{(l)}_{m'm}(G) = \frac{1}{|G|} \sum_{g \in G} D^{(l)}_{m'm}(g),

where :math:`G` is a group of symmetry operations, and :math:`|G|` is the order (number
of elements) of the group :math:`G`. Notice that this formula should be carried out per
:math:`l`, meaning that for each :math:`l` we should expect to have a different matrix
for each operation and group action will be the sum of these matrices. Effectively,
:math:`l` plays the role of the size of the basis sets (of spherical harmonics). So we
shall have :math:`l` matrices for each operation in the group, and :math:`l` matrices
for group action.

Symmetry Point groups
~~~~~~~~~~~~~~~~~~~~~

Infinitely many point groups exist. Point groups are divided into categories according
to the elements they contain: Cyclic groups (starting with Schoenflies symbol C) which
contain operations related to a rotation of a given degree n, rotoreflection groups (S)
which contain rotoreflection operations, Dihedral groups (D) which contain operations
related to rotation of a given degree n and reflection across a plane perpendicular to
the rotation axis, and Cubic/polyhedral groups (O, T, I) which contain symmetry
operations related to important polyhedra in 3D space. We give an overview of important
point groups for materials science and crystallography below, with some
remarks on notation and nomenclature.

With :math:`\hat{\sigma}_h` we label the reflection which is perpendicular (orthogonal)
to the principal symmetry axis. On the other hand :math:`\hat{\sigma}_v` is the
reflection which is parallel to the principal symmetry axis. There are multiple choices
one can make with parallel reflection - it could be in :math:`zx` or :math:`zy` plane.
With :math:`\hat{\sigma}_d` we usually label reflections parallel to the principal axis
that are not :math:`zx` or :math:`zy`.

The group operations are taken from the following `link
<http://symmetry.constructor.university/cgi-bin/group.cgi?group=1>`_. We follow the
nomenclature found in :cite:`ezra` and :cite:`Altmann_semidirect`. In addition to that,
we shall adopt a nomenclature in which :math:`\hat{\sigma}_h = \hat{\sigma}_{xy}` is the
only horizontal reflection plane, while :math:`\hat{\sigma}_{v}` can be any reflection
plane containing principal axis of symmetry in :math:`z` direction. Note that some other
sources (such as :cite:`ezra`) would for some of these reflection planes use
:math:`\hat{\sigma}^{'}`. The designation :math:`\hat{\sigma}_d` denotes a subset of
reflections :math:`\hat{\sigma}_{v}` which also bisect the angle between the twofold
axes perpendicular to the principal symmetry axis(:math:`z`). We opt not to use the
designation :math:`\hat{\sigma}_d`. The definitions for specific operations are also
given `here 
<https://web.archive.org/web/20120813130005/http://newton.ex.ac.uk/research/qsystems/people/goss/symmetry/CharacterTables.html>`_. 

Many operations in the table contain a power. The power is to be read as applying the
same operation multiple times. For example :math:`{\hat{C}_2}^2` applies
:math:`\hat{C}_2` operation twice. The elements of groups :math:`S_n` for odd values of
:math:`n` are also given in :cite:`drago1992`.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Point Group
     - Symmetry Operations
   * - :math:`C_1`
     - :math:`\hat{E}`
   * - :math:`C_s`
     - :math:`\hat{E}`, :math:`\hat{\sigma}_v`
   * - :math:`C_h`
     - :math:`\hat{E}`, :math:`\hat{\sigma}_h`
   * - :math:`C_i`
     - :math:`\hat{E}`, :math:`\hat{i}`
   * - :math:`C_n`
     - :math:`\hat{E}`, :math:`\hat{C}_n`, :math:`{\hat{C}_n}^2`, ... :math:`{\hat{C}_n}^{n-1}`
   * - :math:`C_{nh}`, :math:`n` is even
     - :math:`\hat{E}`, :math:`\hat{C}_n`, :math:`{\hat{C}_n}^2`, ... :math:`{\hat{C}_n}^{n-1}`, :math:`\hat{\sigma}_h`, :math:`\hat{S}_n`, :math:`{\hat{S}_n}^3`, ... :math:`{\hat{S}_n}^{n-1}`
   * - :math:`C_{nh}`, :math:`n` is odd
     - :math:`\hat{E}`, :math:`\hat{C}_n`, :math:`{\hat{C}_n}^2`, ... :math:`{\hat{C}_n}^{n-1}`, :math:`\hat{\sigma}_h`, :math:`\hat{S}_n`, :math:`{\hat{S}_n}^3`, ... :math:`{\hat{S}_n}^{2n-1}`
   * - :math:`C_{nv}`
     - :math:`\hat{E}`, :math:`\hat{C}_n`, :math:`{\hat{C}_n}^2`, ... :math:`{\hat{C}_n}^{n-1}`, :math:`n \hat{\sigma}_v`
   * - :math:`D_n`
     - :math:`\hat{E}`, :math:`\hat{C}_n`, :math:`{\hat{C}_n}^2`, ... :math:`{\hat{C}_n}^{n-1}`, :math:`n \hat{C}_2^{'}` 
   * - :math:`D_{nh}`
     - :math:`\hat{E}`, :math:`\hat{C}_n`, :math:`{\hat{C}_n}^2`, ... :math:`{\hat{C}_n}^{n-1}`, :math:`n \hat{C}_2^{'}`, :math:`\hat{\sigma}_h`, :math:`\hat{S}_n`, :math:`{\hat{S}_n}^3`, ... :math:`{\hat{S}_n}^{n-1}`, :math:`n\hat{\sigma}_v`
   * - :math:`D_{nd}` (sometimes called :math:`D_{nv}`)
     - :math:`\hat{E}`, :math:`\hat{C}_n`, :math:`{\hat{C}_n}^2`, ... :math:`{\hat{C}_n}^{n-1}`, :math:`n \hat{C}_2^{'}`, :math:`\hat{S}_{2n}`, :math:`{\hat{S}_{2n}}^3`, ... :math:`{\hat{S}_{2n}}^{2n-1}`, :math:`n\hat{\sigma}_v`
   * - :math:`S_{n}`, :math:`n` is even
     - :math:`\hat{E}`, :math:`\hat{S}_{n}`, :math:`{\hat{S}_{n}}^2`, ... :math:`{\hat{S}_{n}}^{n-1}`
   * - :math:`S_{n}`, :math:`n` is odd
     - :math:`\hat{E}`, :math:`\hat{S}_{n}`, :math:`{\hat{S}_{n}}^2`, ... :math:`{\hat{S}_{n}}^{2n-1}`
   * - :math:`T`
     - :math:`\hat{E}`, :math:`4 \hat{C}_3`, :math:`4 {\hat{C}_3}^2`, :math:`3 \hat{C}_2`
   * - :math:`T_h`
     - :math:`\hat{E}`, :math:`4 \hat{C}_3`, :math:`4 {\hat{C}_3}^2`, :math:`3\hat{C}_2`, :math:`\hat{i}`, :math:`3 \hat{\sigma}_h`, :math:`4 \hat{S}_6`, :math:`4 {\hat{S}_6}^5`
   * - :math:`T_d`
     - :math:`\hat{E}`, :math:`8 \hat{C}_3`, :math:`3 \hat{C}_2`, :math:`6 \hat{\sigma}_v`, :math:`6\hat{S}_4`
   * - :math:`O`
     - :math:`\hat{E}`, :math:`6 \hat{C}_4`, :math:`8 \hat{C}_3`, :math:`9 \hat{C}_2`
   * - :math:`O_h`
     - :math:`\hat{E}`, :math:`6 \hat{C}_4`, :math:`8 \hat{C}_3`, :math:`9 \hat{C}_2`, :math:`3 \hat{\sigma}_h`, :math:`6\hat{\sigma}_v`, :math:`\hat{i}`, :math:`8\hat{S}_6`, :math:`6\hat{S}_4`
   * - :math:`I`
     - :math:`\hat{E}`, :math:`12 \hat{C}_5`, :math:`12 {\hat{C}_5}^2`, :math:`20\hat{C}_3`, :math:`15 \hat{C}_2`
   * - :math:`I_h`
     - :math:`\hat{E}`, :math:`12 \hat{C}_5`, :math:`12 {\hat{C}_5}^2`, :math:`20\hat{C}_3`, :math:`15 \hat{C}_2`, :math:`15\hat{\sigma}_v`, :math:`\hat{i}`, :math:`12\hat{S}_{10}`, :math:`12{\hat{S}_{10}}^3`, :math:`20\hat{S}_6`

Notes on the table:

* :math:`C_{nv}`: each :math:`\hat{\sigma}_v` is a reflection plane containing the
  principal axis of symmetry starting with :math:`\hat{\sigma}_{yz}`, and rest are
  successive rotation of this plane around :math:`z` axis by :math:`\frac{\pi}{n}`.
* All dihedral groups (:math:`D_n`, :math:`D_{nh}`, :math:`D_{nd}`): each
  :math:`\hat{C}_2^{'}` is perpendicular to the principal axis of symmetry starting with
  :math:`\hat{C}_{2x}` and rest are successive rotation of this plane by
  :math:`\frac{2\pi}{n}`. 
* :math:`D_{nh}`: each :math:`\hat{\sigma}_v` is a reflection plane parallel to
  both principal (:math:`z`) and each :math:`\hat{C}_2^{'}` axis.
* :math:`D_{nd}`: each :math:`\hat{\sigma}_d` is a reflection plane parallel to
  the principal axis of symmetry (:math:`z`) and also contains the vector which
  bisects two neighboring :math:`\hat{C}_2^{'}` axes of symmetry.
* All tetrahedral groups (:math:`T`, :math:`T_h`, :math:`T_d`): see
  :cite:`Altmann_WignerD` for specific proper rotations and also see Hurwitz
  quaternions.
* All octahedral groups (:math:`O`, :math:`O_h`): see Lipshitz and Hurwitz quaternions
  for specific proper rotations
* All icosahedral groups (:math:`I`, :math:`I_h`): see Hurwitz and icosian quaternions
  for specific proper rotations

Several point groups from the table above are equivalent. For more information see `this
link <https://en.wikipedia.org/wiki/Schoenflies_notation#Point_groups>`_. In PgOP all
point groups were constructed from their operations given in the above table.

Point group Order Parameter (PgOP)
----------------------------------

Bond order diagrams (BOD)
~~~~~~~~~~~~~~~~~~~~~~~~~


show it on a concrete example for computation.

Talk about PGOP on liquid states.

Talk about PGOP and how it actually works.

Talk about what it actually computes

Bibliography
-------------
.. bibliography::
   :filter: docname in docnames