# HVM-CUDA: an Interaction Combinator evaluator in CUDA.

## Format

An HVM net is a graph with 4 node types:
- $*     ::=$ `ERA`ser node. Has 0 aux ports.
- $(a b) ::=$ ``CON``structor node. Has 2 aux ports.
- ${a b} ::=$ DUPlicator node. Has 2 aux port.
- $@def  ::=$ lazy reference (to a global definition).

Nodes form a tree-like structure in memory. For example:

```
((* x) {x (y y)})
```
Is interpreted as:

```
(CON (CON ERA x) (DUP x (CON y y)))
```

That is, it is a tree with 5 nodes. The lowcase letters are variables. They
represent 2 ends of a wire. So, for example, in the tree above, the first 'x'
is connected to the second 'x'. Variables are always paired.

A program consists of a root tree, plus list of active pairs. Example:

    (a b)
    & (b a) ~ (x (y *))
    & {y x} ~ @foo

The program above has 1 root 2 active trees pair <tree_0> ~ <tree_1>`). An
active pair is two trees that are connected, and can interact. See below.

# Interactions

Active pairs, also called redexes, are reduced via an *interaction rule*.
There are 6 rules, one for each possible combination of node types:

    & * ~ * =>
      (nothing)

    & (a0 a1) ~ * =>
      & a0 ~ *
      & a1 ~ *

    & {a0 a1} ~ * =>
      & a0 ~ *
      & a1 ~ *

    & (a0 a1) ~ (b0 b1) =>
      a0 ~ b0
      a1 ~ b1

    & {a0 a1} ~ {b0 b1} =>
      a0 ~ b0
      a1 ~ b1

    & (a0 a1) ~ {b0 b1} =>
      & a0 ~ {x y}
      & a1 ~ {z w}
      & b0 ~ (x z)
      & b1 ~ (y w)

Note:
- ERA-ERA returns nothing (it is erased).
- ERA-CON and ERA-DUP erase the CON/DUP and propagates the ERA.
- CON-CON and DUP-DUP erase both nodes and link neighbors.
- CON-DUP duplicates both notes, commuting internal wires.

The process of applying these rules repeatedly is Turing Complete.

# Definitions

A top-level definition is just a statically known closed net, also called a
package. It is represented like a program, with a root and linked trees:

    @foo = (a b)
      & @tic ~ (x a)
      & @tac ~ (x b)

The statement above represents a definition, @foo, with the `(a b)` tree as
the root, and two linked trees: `@tic ~ (x a)` and `@tac ~ (x b)`.

Definitions interact by unrolling, or expanding, to their values, whenever
they're part of an active pair. For example:

    ...
    & @foo ~ ((* a) (* a))
    ...

The program above would expand @foo, resulting in:

    ...
    & (a0 b0) ~ ((* a) (* a))
    & @tic ~ (x0 a0)
    & @tac ~ (x0 b0)
    ...

Note that:
1. We replaced the @foo node by the root of the @foo definition.
2. We added each redex of @foo (`@tic ~ ...` and `@tac ~ ...`) to the
program.
3. To avoid name capture, we appended `0` to `x`, `a`, and `b`.
This program would then proceed to reduce the remaining redexes.

Also, as an optimization, whenever a @def interacts with a node not contained
in that definition, it will just copy the @def. For example, consider:

    & @foo ~ {a b}

Normally, the redex above would expand @foo, but there are no DUP nodes in
the @foo's definition. As such, it will, instead, just copy @foo:

    a ~ @foo
    b ~ @foo

Also, as a similar optimization, `@foo ~ ERA` will just erase the ref node.
That's correct, because references are closed terms, which are always fully
collected by erasure interactions, so, there is no need to unroll it.

# Example Reduction

Consider the original program:

    (a b)
    & (b a) ~ (x (y *))
    & {y x} ~ @foo

It has two redexes. HVM is strongly confluent, which means we can reduce
redexes in any order (and even in parallel) without affecting the total
interaction count. Let's start reducing the first active pair. It is a
CON-CON interaction, which reduces to:

    (a b)
    & b ~ x
    & a ~ (y *)
    & {y x} ~ @foo

Now, we must copy @foo, since it doesn't contain DUP nodes:

    (a b)
    & b ~ x
    & a ~ (y *)
    & y ~ @foo
    & x ~ @foo

Now, we just resolve the 'y ~ @foo' and 'x ~ @foo' wirings:

    (a b)
    & b ~ @foo
    & a ~ (@foo *)

Finally, we resolve the 'b ~ (@foo *)' and 'a ~ @foo' wirings:

    ((@foo *) @foo)

This program is in un-expanded normal form, which can still contain
references (because they never were expanded by the usual rules). We
must proceed to expand them and reduce further:

    ((a0 b0) (a1 b1))
      & @tic ~ (x0 a0)
      & @tac ~ (x0 b0)
      & @tic ~ (x1 a1)
      & @tac ~ (x1 b1)

At this point, we must expand @tic and @tac. Suppose they were just:

    @tic = (k k)
    @tac = (k k)

This would expand to:

    ((a0 b0) (a1 b1))
      & (k0 k0) ~ (x0 a0)
      & (k1 k1) ~ (x0 b0)
      & (k2 k2) ~ (x1 a1)
      & (k3 k3) ~ (x1 b1)

Let's now perform all these CON-CON interactions in parallel:

    ((a0 b0) (a1 b1))
      & k0 ~ x0
      & k0 ~ a0
      & k1 ~ x0
      & k1 ~ b0
      & k2 ~ x1
      & k2 ~ a1
      & k3 ~ x1
      & k3 ~ b1

Let's now resolve the kN wires:

    ((a0 b0) (a1 b1))
      & x0 ~ a0
      & x0 ~ b0
      & x1 ~ a1
      & x1 ~ b1

Let's now resolve the xN wires:

    ((a0 b0) (a1 b1))
      & a0 ~ b0
      & a1 ~ b1

Let's now resolve the aN wires:

    ((b0 b0) (b1 b1))

This is the final result. The program is now in normal form.

# Memory Layout

HVM-CUDA represents programs as tuple:

    Program ::= {
      RBAG : [Pair] // pairs (redex bag)
      NODE : [Pair] // pairs (node buffer)
      SUBS : [Port] // ports (substitutions)
    }

A definition is like a program, with varlen buffers:

    Def ::= {
      PAIR : [Pair]
      NODE : [Node]
    }

A Book is a buffer of definitions:

      Book ::= {
        DEFS: [Def]
      }

A Pair consists of two Ports, representing a either a redex or a node:

    Pair ::= (Port, Port)

A Port consists of a tag and a value:

    Port ::= 3-bit tag + 14-bit val

There are 9 Tags:

    Tag ::=
      | ERA ::= an erasure ndoe
      | CON ::= the CON node
      | DUP ::= the DUP node
      | VAR ::= a variable
      | REF ::= a reference
      | NUM ::= numeric literal
      | OP2 ::= numeric binary op
      | OP1 ::= numeric unary op
      | SWI ::= numeric switch

Since 2 bits can only represent 4 tags, we represent ERA as DUP(0), and leave
CON(0) for empty spaces. That means the first index of the NODE buffer can't
be addressed, so we use it to store the root port.

The RBAG buffer is split into 4 chunks of 1536 elems, 1 for each interaction:
- Collection   : ERA-CON and ERA-DUP interactions - indices 0x0000 to 0x05FF
- Annihilation : CON-CON and DUP-DUP interactions - indices 0x0600 to 0x0BFF
- Commutation  : CON-DUP interaction              - indices 0x1200 to 0x17FF
- Dereference  : REF interactions                 - indices 0x1800 to 0x1DFF
This allows threads to perform the same interaction, reducing divergence.
Note that the ERA-ERA rule isn't included, because it is skipped enirely.

## Memory Layout Example

Consider, again, the following program:

    (a b)
    & (b a) ~ (x (y *))
    & {y x} ~ @foo

In memory, it could be represented as, for example:

- RBAG | FST-TREE | SND-TREE
- ---- | -------- | --------
- 0800 | CON 0001 | CON 0002 // '& (b a) ~ (x (y *))'
- 1800 | DUP 0005 | REF 0000 // '& {x y} ~ @foo'
- ---- | -------- | --------
- NODE | PORT-1   | PORT-2
- ---- | -------- | --------
- 0000 | CON 0001 |          // points to root node
- 0001 | VAR 0000 | VAR 0001 // '(a b)' node (root)
- 0002 | VAR 0001 | VAR 0000 // '(b a)' node
- 0003 | VAR 0002 | CON 0004 // '(x (y *))' node
- 0004 | VAR 0003 | DUP 0000 // '(y *)' node
- 0005 | VAR 0003 | VAR 0002 // '{y x}' node
- ---- | -------- | --------

Notes:
- There are 2 RBAG entries, to represent each redexes.
- There are 5 NODE entries, to represent each CON/DUP node.
- We store the root on PORT-1 of NODE 0000.
- The RBAG buffer is partitioned:
  - The fst redex is a CON-CON, stored on partition 0x0800.
  - The snd redex is a DUP-CON, stored on partition 0x1800.
- VARs always come in pairs. Notice that each VAR appears twice.

# The Interaction Kernel

The heart of HVM's reduction engine is the interaction kernel, which is
executed by 256 to 1024 threads in parallel in a single SM (multprocessor),
depending on the GPU configuration. These are grouped in "squads" of 4
threads each. So, for example, if we launch 256 threads per block, then we'll
have 128 squads. This is done because interaction rules can be further be
broken down in 4 sequential fragments. This increases the maximum theoretical
speedup by 4x, per Amdahl's law!

The interaction kernel works as follows:
