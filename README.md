# anjl - A neighbour-joining library for Python :angel:

`anjl` is a Python package providing implementations of the
[neighbour-joining
algorithm](https://en.wikipedia.org/wiki/Neighbor_joining) of Saitou
and Nei and some associated utilities.

## Installation

```
pip install anjl
```

## Usage

```python
import anjl
```

### Canonical neighbour-joining implementation

```python
help(anjl.canonical_nj)
```

### Rapid neighbour-joining implementation

```python
help(anjl.rapid_nj)
```

### Plot a tree using the equal-angles layout

```python
help(anjl.plot_equal_angles)
```

## About

There are implementations of neighbour-joining available in
[BioPython](https://biopython.org/docs/latest/api/Bio.Phylo.TreeConstruction.html#Bio.Phylo.TreeConstruction.DistanceTreeConstructor),
[scikit-bio](https://scikit.bio/docs/dev/generated/skbio.tree.nj.html)
and
[biotite](https://www.biotite-python.org/latest/apidoc/biotite.sequence.phylo.neighbor_joining.html),
but they are relatively slow for larger numbers of nodes. I created
this package to provide faster implementations for use in population
genomics.

Bug reports, suggestions and pull requests are welcome but I make no promises
regarding support, please be patient and understanding! 🌻🌼🌸
