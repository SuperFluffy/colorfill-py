# A quick and dirty version of the Flood-It game

This is a solver of the Flood-It game. Some details can be found in [Clifford et al's] paper on its complexity.

This implementation relies heavily on `scikit-image` for image
segmentation and `networkx` for path finding. Other than that, nothing
fancy is going on.

There are two strategies implemented:

+ `greedy`: the next move colors the largest adjacent area
+ `smart`: adjacent areas are translated into an adjacency graph; after
  obtaining the longest path possible path from the root node to any of the
  other nodes in the graph, one step along that path is taken, areas are
  recolored, nodes merged, and a new longest path is constructed.

Both strategies progress until the entire board is colored.

The `smart` strategy uses `dijkstra`'s algorithm internally, but the weights
should probably be tweaked to fit the problem better.

[Clifford et al's]: https://arxiv.org/pdf/1001.4420.pdf

## Dependencies

The dependencies are specified in `Pipfile` and `Pipfile.lock`.

Additionally, you might have install `tk`. That's a requirement for
`matplotlib` to run, at least on Linux. It looks like on MacOSX the default
backend for `matplotlib` should work out of the box. Because `tk` bindings
are built into Python's stdlib, simply having the necessary system libraries
available should work.

On Arch Linux:

```bash
$ pacman -S tk
```

## Usage

Install with `pipenv`, activate the shell, and run:

```bash
$ pipenv install
# This might take a while
$ pipenv shell
$ ./colorfill.py -h
usage: colorfill.py [-h] -n SIZE -c COLORS [-s {greedy,smart}] [-f]

Simulate playing the Flood-It game. Click to advance

optional arguments:
  -h, --help            show this help message and exit
  -n SIZE, --size SIZE  The number n of tiles along a side of the n×n grid.
  -c COLORS, --colors COLORS
                        The number of colors in the game
  -s {greedy,smart}, --strategy {greedy,smart}
                        The strategy to be used; default: smart
  -f, --fixed           Used a fixed seed. This way, the board will always
                        look the same.

$ ./colorfill.py -n 10 -c 4
```

This should open a window with a colored grid. Click to advance.
