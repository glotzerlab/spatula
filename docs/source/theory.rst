======
Theory
======

This section will contain the theoretical background of the code. It will be
divided into several subsections, each of which will contain a brief
introduction to the topic, followed by a more detailed explanation of the
relevant concepts.

Introduction
------------

PgOP's main use case is to determine if the local structure around a selected point in
space (which can belong to a particle location, or not) is symmetric with some specific
symmetry in mind. Symmetry is defined as a binary relation between two objects that are
the same under some transformation. Some objects or points in space or systems can
either have some symmetry or not. PgOP removes this binary distinction and instead gives
a continuous measure of how symmetric the object or set of points is with respect to
some symmetry. Such approach can be incredibly useful when studying for example the
local structure of a crystal as it is formed.

Symmetry is a fundamental concept in physics and mathematics. It is a property of an
object that remains unchanged under some transformation. For example, a square has
rotational symmetry of order 4, because it looks the same after a 90 degree rotation.
Symmetry can be described in terms of a group of transformations that leave the object
invariant. This group is called the symmetry group of the object. There are several
types of symmetry operations that are often encountered in physics, such as rotations,
reflections, and translations. These operations can be combined to form more complex 
symmetry operations. Below we overview some of the most common types of symmetry used in
physics and crystallography that are relevant for the PgOP code.

Crystals
--------

Crystals are condensed phase of matter that have long range order. Main, defining
characteristic of a crystal is that it has translational symmetry. A main consequence of
this fact is that the crystal exhibits so-called Bragg peaks in its diffraction pattern.
Crystals are often described using unit cells, due to possession of translational
symmetry, a repeating set of particles in space that can be used to describe the whole
crystal. Relations between these translation vectors allow us to categorize crystals
into 7 crystal systems. 

Particles inside the unit cell often sit in such relations that they obey additional
symmetries. Every crystal has a set of symmetries associated with it stemming from the
translational vectors which define the unit cell and additional optional symmetries.

Crystalline symmetries
~~~~~~~~~~~~~~~~~~~~~~

There are several types of symmetries that are often encountered in crystals. Each
crystal has a point group (of the crystal) as well as a space group associated with it.
The point group of a crystal is defined with respect to the center of the crystal (also
called general position). This position has no equivalent positions, and lies in
position which has full point group symmetry of the crystal. Each
crystallographic point group is compatible with a set of crystal systems and space
groups. The symmetry of a point group leaves a set of particles in this crystal
invariant under the symmetry operations of the point group. Translations are not
considered in the point group. Combining the point group with the translational symmetry
of the crystal gives the space group of the crystal. 

Interestingly, the particles in the
unit cell of the crystal can have different symmetries associated with them. These are
dependent on the position of the particle in the unit cell and are called Wyckoff sites.
The symmetry of the Wyckoff site (called site symmetry) is a subgroup of the point group
of the crystal. This is so because if Wyckoff site lies on a certain symmetry element,
it does not transform according to that symmetry element, which reduces the number of
equivalent positions, such site can generate. Note that this symmetry is NOT the local
point group symmetry of a point in space associated with Wyckoff site position. Each 
crystalline unit cell with a given space/point group will have some "special" sites in
the unit cell. 

However, due to high degree of order particles occupying positions other than general
position (they could be special positions as well) will also often have certain point
group symmetries. These are not reported in crystallographic tables. The point group
symmetry of such a point is not necessarily the same or even related to the point group
of the crystal. The particle in the general position will have the point group symmetry
of the crystal. PgOP enables us to measure continuously how symmetric a particle is with
respect to some symmetry operation throughout the trajectory.

Symmetry operations in point groups
-----------------------------------

There are two main types of symmetry operations associated with point groups: rotations
and reflections. Inversion can be considered as a special case of reflection, while
rotoreflections can be constructed using combination of rotations and inversions. In
space groups in addition to the above mentioned symmetry elements we also have
translational symmetry. This symmetry can be combined with elements of point group
symmetry to obtain new unique type of symmetry operations such as screw axes (rotation +
translation) and glide planes (reflections + translations).

Rotations are defined by the axis of rotation and the angle of rotation. Many
representations of rotations exist and here we give several common ones. The most common
representation of rotations is the rotation matrix. The rotation matrix is a 3x3 matrix
that rotates a vector in 3D space by a given angle around a given axis. The formula in
Cartesian representation is given by :math:`\hat{C}_{2\pi/\theta}=\hat{R}(\theta,
\mathbf{u})`: 

.. math::
    \begin{pmatrix}
    \cos \theta + u_x^2 (1 - \cos \theta) & u_x u_y (1 - \cos \theta) - u_z 
    \sin \theta & u_x u_z (1 - \cos \theta) + u_y \sin \theta \\
    u_y u_x (1 - \cos \theta) + u_z \sin \theta & \cos \theta + u_y^2 (1 - \cos \theta)
     & u_y u_z (1 - \cos \theta) - u_x \sin \theta \\
    u_z u_x (1 - \cos \theta) - u_y \sin \theta & u_z u_y (1 - \cos \theta) + u_x 
    \sin \theta & \cos \theta + u_z^2 (1 - \cos \theta)
    \end{pmatrix}

where :math:`\theta` is the angle of rotation and :math:`\mathbf{u}` is the axis of
rotation in form of a unit vector. All basic operators are denoted by a hat. We shall,
moving forward always consider that the principal axis of symmetry (one with highest
symmetry order) is along the :math:`z` axis.

Another common representation of rotations is the Euler angles. The Euler angles are a
set of three angles that describe the orientation of a rigid body in 3D space. The Euler
angles can also be used to construct the rotation matrix, but the formula depends on the
convention chosen. The Euler angles are not unique, meaning that there are many sets of
Euler angles that describe the same rotation. The Euler angles also suffer from the
so-called `gimbal lock problem <https://en.wikipedia.org/wiki/Gimbal_lock>`_. `The
formula <https://en.wikipedia.org/wiki/Rotation_matrix>`_ for rotation matrix from Euler
angles can be obtained in a given convention by performing 3 successive matrix
multiplications of the rotation matrices for each angle.

Another common and useful representation are `quaternions
<https://en.wikipedia.org/wiki/Quaternion>`_. Quaternions can be used to represent
rotations in 3D space and live on the unit hypersphere in 4D space. Quaternions are
useful because they do not suffer from the gimbal lock problem and are more
computationally efficient than the rotation matrix.

Reflections are defined by the plane of reflection. Reflections are a type of symmetry
operation that flips the object across the plane of reflection. Reflections cannot be
represented as rotations, or combination of rotations in a general sense. Reflections
can be represented as inversion followed by a rotation of 180 degrees
:cite:`Altmann_WignerD`:

.. math::
    \hat{\sigma}_{xy} = \hat{i} \hat{C}_{2z}
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
reflection. Thus, by definition, we can write:

.. math::
    \hat{S}_n = \hat{\sigma}_h {\hat{C}_n} = \hat{\sigma}_{xy} {\hat{C}_n}

where :math:`\hat{\sigma}_h=\hat{\sigma}_{xy}` is the reflection operator perpendicular
to the axis of rotation (:math:`z`).

The alternatively, easier way to write such operation is to first apply rotation by an
angle :math:`\frac{2\pi}{n}`, followed by the inversion :cite:`Altmann_WignerD`: 

.. math::
    \hat{S}_n = \hat{i} {\hat{C}_n}^{n-1} \quad \text{for } n \text{ even} \\

    \hat{S}_n = \hat{i} {\hat{C}_{2n}}^{n-1} \quad \text{for } n \text{ odd}

We also often consider subsequent application of the same operation. This is usually
written in the form :math:`\hat{S}_n^k` where :math:`k` is the number of times the
operation is applied. Following the above relations we can derive several useful
relations for powers of rotoreflections:

.. math::
  (\hat{S}_n)^n &= \hat{E} \quad \text{for } n \text{ even} \\
  (\hat{S}_n)^{2n} &= \hat{E} \quad \text{for } n \text{ odd} \\
  (\hat{S}_n)^n &= \hat{\sigma}_h \quad \text{for } n \text{ odd} \\
  (\hat{S}_n)^m &= (\hat{C}_n)^m \quad \text{for } m \text{ even} \\
  (\hat{S}_n)^m &= \hat{\sigma}_h (\hat{C}_n)^{m} \quad \text{for } m \text{ odd} \\
  (\hat{S}_n)^{m+n} &= \hat{\sigma}_h (\hat{C}_n)^{m} \quad \text{for } m \text{ even and } n \text{ odd}  \\
  (\hat{S}_n)^m &= \hat{\sigma}_h (\hat{C}_n)^{m-n} \quad \text{for } m \text{ odd and } m>n, n \text{ odd}  \\


Group theory
------------

In group theory, sets with operation under certain constraints (operation must be
associative, and has identity element, and every element of the set has inverse) are
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
PGOP and choice for it will become apparent later. For now, we focus our attention on
how to construct these matrices for basic operations.

A single Wigner :math:`D` matrix is defined for a given symmetry operation and a given l, which
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


Let's turn now our attention to Wigner :math:`D` matrices for rotations. The Wigner
:math:`D` matrix for 
a general rotation from Euler angles in zyz convention is given by the formula
:cite:`Altmann_WignerD`: 

.. math::
    D^{(l)}_{m'm}(\hat{R}\left(\alpha, \beta, \gamma\right)) = 
    C_{m'm} e^{im\alpha} e^{im'\gamma} \sqrt{\left(l+m\right)! \left(l-m\right)! 
    \left(l+m'\right)! \left(l-m'\right)! } S^{(l)}_{m'm}(\beta)

where :math:`C_{m'm}` is:

.. math::
    C_{m'm}=i^{\left|m'\right|+m'} i^{\left|m\right|+m}

and where :math:`S^{(l)}_{m'm}(\beta)` is:

.. math::
    S^{(l)}_{m'm}(\beta) = \sum_{k=\max(0,m-m')}^{\min(l-m',l+m)}
    \frac{(-1)^k \cos^{2l+m-m'-2k} \left(\frac{\beta}{2}\right) \sin^{2k+m'-m} 
    \left(\frac{\beta}{2}\right)}{(l-m'-k)! (l+m-k)! k! (k-m+m')!} 

This expression can be simplified for values of :math:`\beta` equal to :math:`0`,
:math:`\frac{\pi}{2}`, and :math:`\pi`, which are all the rotations relevant for
crystallographic point groups. 

For :math:`\beta=0`, the Wigner :math:`D` matrix is given by :cite:`Altmann_WignerD`:

.. math::
    D^{(l)}_{m'm}(\hat{R}\left(\alpha, 0, \gamma\right)) = 
    e^{im\alpha} e^{im'\gamma} \delta_{m',m}

where :math:`\delta_{m',m}` is the Kronecker delta.

For :math:`\beta=\frac{\pi}{2}`, the Wigner :math:`D` matrix is given by the general
expression with simplified :math:`S^{(l)}_{m'm}(\beta)` :cite:`Altmann_WignerD` :

.. math::
    S^{(l)}_{m'm}\left(\frac{\pi}{2}\right) = 
    2^{-l} \sum_{k=\max(0, m - m')}^{\min(l - m', l + m)} 
    \frac{(-1)^k}{(l - m' - k)! (l + m - k)! k! (k - m + m')!}

For :math:`\beta=\pi`, the Wigner :math:`D` matrix is given by :cite:`Altmann_WignerD` :

.. math::
    D^{(l)}_{m'm}(\hat{R}\left(\alpha, \pi, \gamma\right)) = 
    (-1)^l e^{im\alpha} e^{im'\gamma} \delta_{m',-m}

From here we can derive Wigner :math:`D` matrices for useful rotations such as a general
:math:`\hat{C}_n` rotation around the :math:`z` axis by evaluating
:math:`D^{(l)}_{m'm}(\hat{R}(\frac{2\pi}{n}, 0, 0))`:

.. math::
    D^{(l)}_{m'm}(\hat{C}_n) = e^{im\frac{2\pi}{n}} \delta_{m',m}.

Two fold rotation around the :math:`y` axis is given by :math:`D^{(l)}_{m'm}(\hat{R}(0,
\pi, 0))`:

.. math::
    D^{(l)}_{m'm}(\hat{C}_{2y}) = (-1)^l \delta_{m',-m}

while two fold rotation around the :math:`x` axis is given by
:math:`D^{(l)}_{m'm}(\hat{R}(\pi,\pi,0))`:

.. math::
    D^{(l)}_{m'm}(\hat{C}_{2x}) = (-1)^{(l+m)} \delta_{m',-m}

Inversion operation is given by the formula :cite:`Altmann_WignerD, engel2021point`:

.. math::
    D^{(l)}_{m'm}(\hat{i}) = (-1)^l \delta_{m',m} 

Another important operation is the identity operation, which is given by the formula:

.. math::
    D^{(l)}_{m'm}(\hat{E}) = \delta_{m',m}

From these elementary formulas we can construct several Wigner :math:`D` matrices for
other useful operations such as reflections and rotoreflections. We have already shown
that reflections can be constructed from inversions and rotations using the formula
:math:`\hat{\sigma} = \hat{i} \hat{C}_2`. To compute the resulting :math:`D` matrix for
the reflection operation we simply perform matrix multiplication to obtain the matrix
representation of the new operator. Let's consider formulas for several useful
situations, when the reflection plane lies in the :math:`xy`, :math:`xz`, and :math:`yz`
plane. The Wigner :math:`D` matrix for reflection across the :math:`xy` plane is given
by :cite:`Altmann_WignerD`:

.. math::
    D^{(l)}_{m'm}(\hat{\sigma}_{xy}) = D^{(l)}_{m'm}(\hat{i}) \times 
    D^{(l)}_{m'm}(\hat{C}_{2z}) = (-1)^{m+l} \delta_{m',m}

The Wigner :math:`D` matrix for reflection across the :math:`xz` plane is given by:

.. math::
    D^{(l)}_{m'm}(\hat{\sigma}_{xz}) = D^{(l)}_{m'm}(\hat{i}) \times 
    D^{(l)}_{m'm}(\hat{C}_{2y}) =  \delta_{m',-m}

The Wigner :math:`D` matrix for reflection across the :math:`yz` plane is given by:

.. math::
    D^{(l)}_{m'm}(\hat{\sigma}_{yz}) = D^{(l)}_{m'm}(\hat{i}) \times 
    D^{(l)}_{m'm}(\hat{C}_{2x}) = (-1)^m \delta_{m',-m}

Lastly we attempt to derive formulas for rotoreflections. We have already shown that
rotoreflections can be constructed from inversions and rotations using the formula
:math:`\hat{S}_n = \hat{\sigma}_{xy} \hat{C}_n`. The resulting :math:`D` matrix for
rotoreflection operation is given by:

.. math::
    D^{(l)}_{m'm}(\hat{S}_{n}) = D^{(l)}_{m'm}(\hat{\sigma}_{xy}) \times D^{(l)}_{m'm}(\hat{C}_n) = 
    (-1)^{m+l} e^{im\frac{2\pi}{n}} \delta_{m',m}.

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

Infinitely many point groups exist, but only 32 of them are allowed by `crystallographic
restriction theorem
<https://en.wikipedia.org/wiki/Crystallographic_restriction_theorem>`_. Point groups can
loosely be divided into categories 
according to the elements they contain: Cyclic groups (starting with Schoenflies symbol
C) which contain operations related to a rotation of a given degree n, rotoreflection
groups (S) which contain rotoreflection operations, Dihedral groups (D) which contain
operations related to rotation of a given degree n and reflection across a plane
perpendicular to the rotation axis, and Cubic/polyhedral groups (O, T, I) which contain
symmetry operations related to important polyhedra in 3D space. We give an overview of
important point groups for materials science and crystallography below, with some
remarks on notation and nomenclature.

With :math:`\hat{\sigma}_h` we label the reflection which is perpendicular (orthogonal)
to the principal symmetry axis. On the other hand :math:`\hat{\sigma}_v` is the
reflection which is parallel to the principal symmetry axis. There are multiple choices
one can make with parallel reflection - it could be in :math:`zx` or :math:`zy` plane.
With :math:`\hat{\sigma}_d` we usually label reflections parallel to the principal axis
that are not :math:`zx` or :math:`zy`.

The group operations are taken from the following `link
<https://web.archive.org/web/20120717074206/http://newton.ex.ac.uk/research/qsystems/people/goss/symmetry/CC_All.html>`_.
Note that several errors are present, such as operations for :math:`C_{5h}`. Also note
that :math:`\hat{C}_s` in :cite:`ezra` contains :math:`\hat{\sigma}_{yz} =\hat{\sigma}_v`
and on the web page it conatins :math:`\hat{\sigma}_h`. We follow the nomenclature found
in :cite:`ezra` and :cite:`Altmann_semidirect`. In addition to that, we shall adopt a
nomenclature in which :math:`\hat{\sigma}_h = \hat{\sigma}_{xy}` is the only horizontal
reflection plane, while :math:`\hat{\sigma}_{v}` can be any reflection plane containing
principal axis of symmetry in :math:`z` direction. Note that some other sources (such as
:cite:`ezra`) would for some of these reflection planes use :math:`\hat{\sigma}^{'}`.
The designation :math:`\hat{\sigma}_d` denotes reflection plane which does not contain
principal symmetry axis (:math:`z`), and is not perpendicular to it. The definitions for
specific operations are also given `here 
<https://web.archive.org/web/20120813130005/http://newton.ex.ac.uk/research/qsystems/people/goss/symmetry/CharacterTables.html>`_. 

Many operations in the table contain a power. The power is to be read as applying the
same operation multiple times. For example :math:`{\hat{C}_2}^2` applies
:math:`\hat{C}_2` operation twice.

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
     - :math:`\hat{E}`, :math:`\hat{C}_n`, :math:`{\hat{C}_n}^2`, ... :math:`{\hat{C}_n}^{n-1}`, :math:`n \hat{C}_2^{'}`, :math:`\hat{\sigma}_h`, :math:`\hat{S}_n`, :math:`{\hat{S}_n}^3`, ... :math:`{\hat{S}_n}^{n-1}`, :math:`10\hat{\sigma}_v`
   * - :math:`D_{nd}` (sometimes called :math:`D_{nv}`)
     - :math:`\hat{E}`, :math:`\hat{C}_n`, :math:`{\hat{C}_n}^2`, ... :math:`{\hat{C}_n}^{n-1}`, :math:`n \hat{C}_2^{'}`, :math:`n\hat{\sigma}_v`, :math:`\hat{S}_{2n}`, :math:`{\hat{S}_{2n}}^3`, ... :math:`{\hat{S}_{2n}}^{2n-1}`
   * - :math:`S_{n}`, :math:`n` is even
     - :math:`\hat{E}`, :math:`\hat{S}_{n}`, :math:`{\hat{S}_{n}}^2`, ... :math:`{\hat{S}_{n}}^{n-1}`
   * - :math:`S_{n}`, :math:`n` is odd
     - :math:`\hat{E}`, :math:`\hat{S}_{n}`, :math:`{\hat{S}_{n}}^2`, ... :math:`{\hat{S}_{n}}^{2n-1}`
   * - :math:`T`
     - :math:`\hat{E}`, :math:`4 \hat{C}_3`, :math:`4 {\hat{C}_3}^2`, :math:`3 \hat{C}_2`
   * - :math:`T_h`
     - :math:`\hat{E}`, :math:`4 \hat{C}_3`, :math:`4 {\hat{C}_3}^2`, :math:`3\hat{C}_2`, :math:`\hat{i}`, :math:`3 \hat{\sigma}_h`, :math:`4 \hat{S}_6`, :math:`4 {\hat{S}_6}^5`
   * - :math:`T_d`
     - :math:`\hat{E}`, :math:`8 \hat{C}_3`, :math:`3 \hat{C}_2`, :math:`6 \hat{\sigma}_d`, :math:`6\hat{S}_4`
   * - :math:`O`
     - :math:`\hat{E}`, :math:`6 \hat{C}_4`, :math:`8 \hat{C}_3`, :math:`9 \hat{C}_2`
   * - :math:`O_h`
     - :math:`\hat{E}`, :math:`6 \hat{C}_4`, :math:`8 \hat{C}_3`, :math:`9 \hat{C}_2`, :math:`3 \hat{\sigma}_h`, :math:`6\hat{\sigma}_d`, :math:`\hat{i}`, :math:`8\hat{S}_6`, :math:`6\hat{S}_4`
   * - :math:`I`
     - :math:`\hat{E}`, :math:`12 \hat{C}_5`, :math:`12 {\hat{C}_5}^2`, :math:`20\hat{C}_3`, :math:`15 \hat{C}_2`
   * - :math:`I_h`
     - :math:`\hat{E}`, :math:`12 \hat{C}_5`, :math:`12 {\hat{C}_5}^2`, :math:`20\hat{C}_3`, :math:`15 \hat{C}_2`, :math:`15\hat{\sigma}_d`, :math:`\hat{i}`, :math:`12\hat{S}_{10}`, :math:`12{\hat{S}_{10}}^3`, :math:`20\hat{S}_6`

Notes on the table:

* :math:`C_{nv}`: each :math:`\hat{\sigma}_v` is a reflection plane containing the
  principal axis of symmetry starting with :math:`\hat{\sigma}_{zx}`, and rest are
  successive rotation of this plane around :math:`z` axis by :math:`\frac{2\pi}{n}`.
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
  :cite:`Altmann_WignerD` for specific proper rotations (see Hurwitz quaternions).
* All octahedral groups (:math:`O`, :math:`O_h`): see Lipshitz and Hurwitz quaternions
  for specific proper rotations
* All icosahedral groups (:math:`I`, :math:`I_h`): see Hurwitz and icosian quaternions
  for specific proper rotations

Group theory: generators
~~~~~~~~~~~~~~~~~~~~~~~~

Instead of providing all the elements, groups can be described in terms of generators
(generating sets). A generating set is a set of symmetry operations which can recover
all the elements of the group by applying any combination of any power of the generator
elements. The standard notation for this approach is to write group elements using curly
brackets and generating sets as pointy brackets. For example, group :math:`C_2 =
{\hat{E}, \hat{C}_2} = \langle\hat{C}_2\rangle`. Thus the generator for the :math:`C_2`
group is just the operation :math:`\hat{C}_2`. This is because we can construct the
identity operation (:math:`\hat{E}`) by applying :math:`\hat{C}_2` twice.


.. list-table::
   :header-rows: 1
   :widths: 5 10

   * - Point Group
     - Group generators
   * - :math:`C_1`
     - :math:`\langle\rangle`
   * - :math:`C_s`
     - :math:`\langle\hat{\sigma}_v\rangle = \langle\hat{\sigma}_{yz}\rangle`
   * - :math:`C_h`
     - :math:`\langle\hat{\sigma}_h\rangle = \langle\hat{\sigma}_{xy}\rangle`
   * - :math:`C_i`
     - :math:`\langle\hat{i}\rangle`
   * - :math:`C_n`
     - :math:`\langle\hat{C}_n\rangle = \langle\hat{C}_{nz}\rangle`
   * - :math:`C_{nh}`
     - :math:`\langle\hat{C}_n, \hat{\sigma}_h\rangle =  \langle\hat{C}_n, \hat{\sigma}_{xy}\rangle`
   * - :math:`C_{nv}`
     - :math:`\langle\hat{C}_n, \hat{\sigma}_v\rangle =  \langle\hat{C}_n, \hat{\sigma}_{yz}\rangle`
   * - :math:`D_n`
     - :math:`\langle\hat{C}_n, \hat{\sigma}_h \hat{\sigma}_v\rangle =  \langle\hat{C}_n, \hat{\sigma}_{xy} \hat{\sigma}_{yz}\rangle`
   * - :math:`D_{nh}` (:math:`D_{nv}`)
     - :math:`\langle\hat{C}_n, \hat{\sigma}_h, \hat{\sigma}_v\rangle =  \langle\hat{C}_n, \hat{\sigma}_{xy}, \hat{\sigma}_{yz}\rangle`
   * - :math:`D_{nd}`
     - :math:`\langle\hat{C}_{2n}\hat{\sigma}_h, \hat{\sigma}_v\rangle =  \langle\hat{C}_{2n} \hat{\sigma}_{xy}, \hat{\sigma}_{yz}\rangle`
   * - :math:`S_{n}`
     - :math:`\langle\hat{S}_{n}\rangle`


Note that for even :math:`n` the group :math:`S_n` can also be generated by the
:math:`S_n = \langle\hat{C}_{n}\hat{\sigma}_h\rangle`. 

In mathematics presentation is used as a way of specifying a group using generators.
Similar nomenclature is used as above with addition of a set of relations. Relations
define how the generators power of the generators can be combined to obtain the identity
element. For example, the group :math:`C_2` can be written as :math:`\langle\hat{C}_2 |
\hat{C}_2^2 = \hat{E}\rangle`. This means that the group :math:`C_2` is generated by the
operation :math:`\hat{C}_2` and the relation is that squaring the operation gives the
identity operation. This notation is useful because allows easy evaluation of group
products.

Group theory operations: direct and semidirect products
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The more complicated groups can usually be constructed from simpler groups using certain
operation specific to groups. First such operation is called direct product. The direct
product of two groups :math:`G` and :math:`H` is a group :math:`G \times H` which is 
constructed by taking all possible pairs of elements from :math:`G` and :math:`H` under
the common operation of those two groups. In case of symmetry point groups, this
operation is composition. In terms of Wigner D matrices, the direct product of two
groups is constructed by taking all possible pairs of Wigner D matrices from the two
groups and multiplying them together:

.. math::
    D^{(l)}_{m'm}(G \times H) =\frac{1}{|G|} \frac{1}{|H|} \sum_{g \in G} \sum_{h \in H} D^{(l)}_{m'm}(g h) = \sum_{m''=-l}^l D^{(l)}_{m'm''}(g) D^{(l)}_{m''m}(h).

Not all point groups can be used in direct
product. The direct product of two point groups is only allowed if the two point groups
commute under the direct product operation. This is only the case if all possible pairs
of operations between the two groups commute :cite:`Altmann_semidirect`:

.. math::
    g_i h_j = h_j g_i,

for all :math:`g_i \in G` and :math:`h_j \in H`. Thus, property of direct product is
that it is commutative. Groups that commute under direct product are abelian groups.
Abelian point groups are cyclic groups or direct products of cyclic groups.

We can also partially loosen the constraint of commutativity by using the semidirect
product if the two groups obey :cite:`Altmann_semidirect`:

.. math::
    Gh_j=h_jG,

for all :math:`h_j \in H`. If the above is true then necessarily the new set
:math:`{gh, g \in G, h \in H}` is also a (point) group. Note that not every combination
of point groups obeys this constraint. The semidirect product of two groups :math:`G`
and :math:`H` is a group :math:`G \rtimes H` which is constructed by taking all possible
pairs of elements from :math:`G` and :math:`H` under the common operation of those two
groups (just like the direct product). The twist is that the operation of :math:`H` on
:math:`G` is not the same as the operation of :math:`G` on :math:`H`. In terms of Wigner
D matrices, the semidirect product of two groups is constructed by taking all possible
pairs of Wigner D matrices from the two groups and multiplying them together:

.. math::
    D^{(l)}_{m'm}(G \rtimes H) =\frac{1}{|G|} \frac{1}{|H|} \sum_{g \in G} \sum_{h \in H} D^{(l)}_{m'm}(g h) = \sum_{m''=-l}^l D^{(l)}_{m'm''}(g) D^{(l)}_{m''m}(h).

The direct product can also be written in terms of group presentations. In general,
given a group :math:`G = \langle S_G | R_G \rangle` and a group :math:`H = \langle S_H |
R_H \rangle`, the direct product of the two groups is given by:

.. math::
    G \times H = \langle S_G \cup S_H | R_G \cup R_H \cup R_P \rangle,

where :math:`R_P` is the set of relations that specify how elements of two groups
commute. Consider group :math:`C_{nh} = \langle \hat{C}_n, \hat{\sigma}_h | \hat{C}_n^n=\hat{E}, \hat{\sigma}_h^2 = \hat{E} \rangle`.
The direct product that gives this group can be written as 
:math:`C_n \times C_s =C_{nh}`.

Point group equivalencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on the group elements, their generators and relations between powers of
rotoreflections and pure reflections, we can conclude that :math:`C_{nh} = S_n` for
every odd value of :math:`n`.

Construction of point groups for PgOP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
accordingt t wikipedia (https://en.wikipedia.org/wiki/Point_groups_in_three_dimensions)

Cnh = Cn x Ch (also Ezra)

Cnv = Cn x {E, sigma_v} (also ezra, in ezra Cs contains sigma_yz, which is sigma_v)

Dn = Cn x {E, sigma_v sigma_h} (also ezra, ezra writes C2y which is equivalent to

sigma_v sigma_h if sigma_v is sigma_yz, which matches ezra notation)

construction used here
It seems Dnh = Dn x Ch = Cnv x Ch = Dn x Ci for even n (gott acheck if sigma_h = i
sometimes?) (this is all from ezra)

(partially agrees with ezra) Cnh = Cn x Ch but only for even n!! !NO because sigma Cn^2 is not Sn^2

for odd n and even m (Sn)^m = E
sigmah C3^2 is NOT S3^2
sigmah C6^2 = S3 

Some useful results of some operations:

.. math::
  (\hat{S}_n)^m &= (\hat{C}_n)^m \quad \text{for } m \text{ even} \\
  (\hat{S}_n)^m &= \hat{\sigma}_h (\hat{C}_n)^{m} \quad \text{for } m \text{ odd} \\
  \hat{\sigma}_h (\hat{C}_n)^{m} &= \hat{C}_n \hat{\sigma}_h(\hat{C}_n)^{m-1} = \hat{C}_n (\hat{S}_n)^{m-1} = \hat{S}_n (\hat{C}_n)^{m-1}  \quad \text{for } m \text{ even} \\


Subgroups of crystallographic point groups and relation to Wyckoff sites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Notice that some point groups contain operations that are also found in other point
groups. In that case we formally say that one group is a subgroup of another group.

Also have a table that lists all subgroups of all crystallographic point groups as well
as a guide how to figure out what subgroup of point group wyckoff site is from
designation such as 24l or similar?

Point group Order Parameter (PgOP)
----------------------------------

Bond order diagrams (BOD)
~~~~~~~~~~~~~~~~~~~~~~~~~


show it on a concrete example for computation.

Talk about PGOP on liquid states.

Talk about PGOP and how it actually works.

Bibliography
-------------
.. bibliography::
   :filter: docname in docnames