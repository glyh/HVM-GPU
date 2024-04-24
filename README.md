## What
This is a repo containing some of my work for the undergratudate thesis "An Optimization on Performance and Scalability of a GPU-based Lock-free Interaction Combinators: HVM". In the thesis I design some techniques to optimize the CUDA kernel for HVM. It is pretty much still a WIP.

## [How](doc/how.md)

## Reference
- Some information by Victor on Twitter [here](https://twitter.com/VictorTaelin/status/1772389464628670705). 
- Adapted from a gist posted by Victor [here](https://gist.github.com/VictorTaelin/a92498156d49ae79ea55eee65944a1b4).

## Route
Eventually, I would like to have this rewritten in Zig. But it is currently a no go for me. For now zig has 0 support for CUDA. It breaks every other release and to catch up people have to rewrite their build.zig constantly. This is not a healthy state of an ecosystem.

For now it's better to keep everything good-old-c.
