.. highlight:: shell

======
Theory
======

This section will contain the theoretical background of the code. It will be
divided into several subsections, each of which will contain a brief
introduction to the topic, followed by a more detailed explanation of the
relevant concepts.

Introduction
------------

PgOP main use case is to determine if the local structure around a selected point in
space (which can belong to a particle location, or not) is symmetric with some specific
symmetry in mind. Symmetry is defined as a binary relation between two objects that are
the same under some transformation. Some objects or points in space or systems can
either have some symmetry or not. PgOP removes this binary distinction and instead gives
a continuous measure of how symmetric the object or set of points is with respect to
some symmetry. Such approach can be incredibly useful when studying for example the
local structure of a crystal as it is formed.

Symmetry
--------

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
Because of this translational symmetry crystals are often described using unit cells, a
repeating set of particles in space that can be used to describe the whole crystal.
Relations between these translation vectors allow us to categorize crystals into 7
crystal systems. 

Particles inside the unit cell often sit in such relations that they obey additional
symmetries. Every crystal has a set of symmetries associated with it stemming from the
translational vectors which define the unit cell and additional optional symmetries.

Crystalline symmetries
~~~~~~~~~~~~~~~~~~~~~~

There are several types of symmetries that are often encountered in crystals. Each
crystal has a point group (of the crystal) as well as a space group associated with it.
The point group of a crystal is defined with respect to the center of the crystal. Each
crystallographic point gruop is compatible with a set of crystal systems and space
groups. The symmetry of a point group leaves a set of particles in this crystal
invariant under the symmetry operations of the point group. Translations are not
considered in the point group. Combining the point group with the translational symmetry
of the crystal gives the space group of the crystal. 

Interestingly, the particles in the
unit cell of the crystal can have different symmetries associated with them. These are
dependent on the position of the particle in the unit cell and are called wyckoff sites.
The symmetry of the wyckoff site (called site symmetry) is a subgroup of the point group
of the crystal. Each 
crystalline unit cell with a given space/point group will have some "special" sites in
the unit cell. These are usually defined by the elements of the point group of the
symmetry, such as the symmetry axis or symmetry plane. The point group symmetry of such
a particle will have a point group symmetry associated with it and this point group will
be different (in general) from the point group of other special sites in the unit cell.
The site symmetry of a wyckoff site is the local point group symmetry associated with
the point in space (unit cell) that this particle occupies and is a measure of the local
environment of the particle. PgOP enables us to measure continuously how symmetric a
particle is with respect to some symmetry operation throughout the trajectory.


Point groups
------------
There are two main types of symmetry operations associated with point groups: rotations
and reflections. Inversion can be considered as a special case of reflection. 

Wigner D matrices
~~~~~~~~~~~~~~~~~

Point group Order Parameter (PgOP)
----------------------------------

Bond order diagrams (BOD)
~~~~~~~~~~~~~~~~~~~~~~~~~


show it on a concrete example for computation.

Talk about PGOP on liquid states.

Talk about PGOP and how it actually works.