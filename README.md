# Rusty-Ramsey

## [Ramsey's Theorem](https://en.wikipedia.org/wiki/Ramsey%27s_theorem)
---
Ramsey's Theorem says that for any *a* and *b* there exists a number *n* such that all graphs of more than *n* vertices contains either a complete *a*-vertex subgraph, or its complement contains a complete *b*-vertex subgraph. This program seeks to recursively enumerate all such graphs for some given *a* and *b*. With that said, it is unlikely to be powerful enough to produce any breakthroughs in the field, and is instead mostly an exercise in writing efficient, multithreaded code to handle large amounts of data in everyone's favorite, *O*(n!) complexity space!

## Terminology
---
A graph is said to be *(n, m) clean* if it contains no complete subgraph of n vertices, and if its complement contains no complete subgraph of m vertices.

A graph *g*'s *extensions* is the set of graphs generated from *g* by adding another vertex and either connecting or not connecting it with each existing vertex. Notably, if *g* has n vertices, the set of *g*'s extensions contains 2^n graphs.

## Algorithm
---
Start with a set of graphs that satisfy your requirement, ie that are (n, m) clean. Then for each graph in that set, generate its extensions and remove those that are not (n, m) clean. Repeat until there are no (n, m) clean extensions. The number of vertices of your largest (n, m) clean graph is the desired Ramsey number.

The problem with this approach is that the number of graphs grows faster than exponentially, which is less than ideal. So the algorithm implemented in the program has the additional step of gathering all extensions with a certain number of vertices, and removing all graphs that are redundant in an isomorphic sense. This of course adds additional complexity, but since this project is mostly an exercise, this is not the end of the world. In the end with some clever tricks that I will go over later, the vast majority of graphs can be checked for isomorphic duplicates relatively instantly. The exception are a handful of particularly troublesome graphs that then take 99% of the computing time.

### Isomorphism Algorithm

The naive approach quickly becomes unusable because it takes *O*(n!) time to check if two graphs are isomorphic, plus there are *O*(n^2) comparisons, and when there are more than a million graphs with 10 vertices, this is outright impossible. The solution I have implemented is to apply labeling algorithms to the graphs. These labeling algorithms assign to each vertex a label. When checking permutations, instead of permuting all vertices, you only need to permute vertices with the same label. Furthermore, this has the advantage of discriminating graphs outright. Two graphs can only be isomorphic if they have the same number of vertices with the same label. This has the effect of vastly reducing the number of comparisons and that number of permutations to check.

The problem is that while the vast majority of graphs are reigned in, there are some that are so regular that my labeling schemes fail to differentiate their vertices. These are the troublesome graphs that take up 99% of the execution time.

## Description of Program
---
Cloning this repo and running 
    
    cargo run --release
    
will start the program. The console output will start to display messages showing what the program is currently doing. It will spin up 4 threads and start finding ramsey graphs. You will note that the program spends most of its time checking new graphs for complete graphs with 4 vertices, as described above.

This will require at least 16 GB of memory to run.