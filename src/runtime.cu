// ./cuda_info.txt//
// ./.hvm.rs//

// 1 million rewrites / second / shading unit
// project 16 billion rewrites / second

// Below is the new CUDA implementation. It has many changes and adaptations.

// HVM-CUDA: an Interaction Combinator evaluator in CUDA.
//
// # Format
//
// An HVM net is a graph with 8 node types:
// - *       ::= ERAser node.
// - #N      ::= NUMber node.
// - @def    ::= REFerence node.
// - x       ::= VARiable node.
// - (a b)   ::= CONstructor node.
// - {a b}   ::= DUPlicator node.
// - <+ a b> ::= OPErator node.
// - ?<a b>  ::= SWItch node.
//
// Nodes form a tree-like structure in memory. For example:
//
//     ((* x) {x (y y)})
//
// Represents a tree with 3 CON nodes, 1 ERA node and 4 VAR nodes.
//
// A program consists of a root tree, plus list of redexes. Example:
//
//     (a b)
//     & (b a) ~ (x (y *))
//     & {y x} ~ @foo
//
// The program above has a root and 2 redexes (in the shape `& A ~ B`).
//
// # Interactions
//
// Redexes are reduced via *interaction rules*:
//
// ## 0. LINK
//
//     a ~ b
//     ------ LINK
//     a ~> b
//
// ## 1. VOID
//
//     * ~ *
//     -----
//     void
//
// ## 2. ERAS
//
//     (A1 A2) ~ *
//     -----------
//     A1 ~ *
//     A2 ~ *
//
// ## 3. ANNI (https://i.imgur.com/ASdOzbg.png)
//
//     (A1 A2) ~ (B1 B2)
//     -----------------
//     A1 ~ B1
//     A2 ~ B2
//
// ## 4. COMM (https://i.imgur.com/gyJrtAF.png)
//
//     (A1 A2) ~ {B1 B2}
//     -----------------
//     A1 ~ {x y}
//     A2 ~ {z w}
//     B1 ~ (x z)
//     B2 ~ (y w)
//
// ## 5. OPER
//
//     <+ A1 A2> ~ #B
//     --------------
//     if A1 is #A:
//       A2 ~ #A+B
//     else:
//       A1 ~ <+ #B A2>
//
// ## 6. SWIT
//
//     ?<A1 A2> ~ #B
//     -------------
//     if B == 0:
//       A1 ~ (A2 *)
//     else:
//       A1 ~ (* (#B-1 A2))
//
// # 7. CALL
//
//     @foo ~ B
//     ---------------
//     deref(@foo) ~ B
//
// # Interaction Table
//
// | A\B |  VAR |  REF |  ERA |  NUM |  CON |  DUP |  OPR |  SWI |
// |-----|------|------|------|------|------|------|------|------|
// | VAR | LINK | LINK | LINK | LINK | LINK | LINK | LINK | LINK |
// | REF | LINK | VOID | VOID | VOID | CALL | CALL | CALL | CALL |
// | ERA | LINK | VOID | VOID | VOID | ERAS | ERAS | ERAS | ERAS |
// | NUM | LINK | VOID | VOID | VOID | ERAS | ERAS | OPER | CASE |
// | CON | LINK | CALL | ERAS | ERAS | ANNI | COMM | COMM | COMM |
// | DUP | LINK | CALL | ERAS | ERAS | COMM | ANNI | COMM | COMM |
// | OPR | LINK | CALL | ERAS | OPER | COMM | COMM | ANNI | COMM |
// | SWI | LINK | CALL | ERAS | CASE | COMM | COMM | COMM | ANNI |
//
// # Definitions
//
// A top-level definition is just a statically known closed net, also called a
// package. It is represented like a program, with a root and linked trees:
//
//     @foo = (a b)
//     & @tic ~ (x a)
//     & @tac ~ (x b)
//
// The statement above represents a definition, @foo, with the `(a b)` tree as
// the root, and two linked trees: `@tic ~ (x a)` and `@tac ~ (x b)`. When a
// REF is part of a redex, it expands to its complete value. For example:
//
//     & @foo ~ ((* a) (* a))
//
// Expands to:
//
//     & (a0 b0) ~ ((* a) (* a))
//     & @tic ~ (x0 a0)
//     & @tac ~ (x0 b0)
//
// As an optimization, `@foo ~ {a b}` and `@foo ~ *` will NOT expand; instead,
// it will copy or erase when it is safe to do so.
//
// # Example Reduction
//
// Consider the first example, which had 2 redexes. HVM is strongly confluent,
// thus, we can reduce them in any order, even in parallel, with no effect on
// the total work done. Below is its complete reduction:
//
//     (a b) & (b a) ~ (x (y *)) & {y x} ~ @foo
//     ----------------------------------------------- ANNI
//     (a b) & b ~ x & a ~ (y *) & {y x} ~ @foo
//     ----------------------------------------------- COMM
//     (a b) & b ~ x & a ~ (y *) & y ~ @foo & x ~ @foo
//     ----------------------------------------------- LINK `y` and `x`
//     (a b) & b ~ @foo & a ~ (@foo *)
//     ----------------------------------------------- LINK `b` and `a`
//     ((@foo *) @foo)
//     ----------------------------------------------- CALL `@foo` (optional)
//     ((a0 b0) (a1 b1))
//     & @tic ~ (x0 a0) & @tac ~ (x0 b0)
//     & @tic ~ (x1 a1) & @tac ~ (x1 b1)
//     ----------------------------------------------- CALL `@tic` and `@tac`
//     ((a0 b0) (a1 b1))
//     & (k0 k0) ~ (x0 a0) & (k1 k1) ~ (x0 b0)
//     & (k2 k2) ~ (x1 a1) & (k3 k3) ~ (x1 b1)
//     ----------------------------------------------- ANNI (many in parallel)
//     ((a0 b0) (a1 b1))
//     & k0 ~ x0 & k0 ~ a0 & k1 ~ x0 & k1 ~ b0
//     & k2 ~ x1 & k2 ~ a1 & k3 ~ x1 & k3 ~ b1
//     ----------------------------------------------- LINK `kN`
//     ((a0 b0) (a1 b1))
//     & x0 ~ a0 & x0 ~ b0 & x1 ~ a1 & x1 ~ b1
//     ----------------------------------------------- LINK `xN`
//     ((a0 b0) (a1 b1)) & a0 ~ b0 & a1 ~ b1
//     ----------------------------------------------- LINK `aN`
//     ((b0 b0) (b1 b1))
//
// # Memory Layout
//
// An HVM-CUDA program includes a redex bag, a node buffer and a vars buffer:
//
//     Program ::= { RBAG: [Pair], NODE: [Pair], VARS: [Port] }
//
// A Pair consists of two Ports, representing a either a redex or a node:
//
//     Pair ::= (Port, Port)
//
// A Port consists of a tag and a value:
//
//     Port ::= 3-bit tag + 13-bit val
//
// There are 8 Tags:
//
//     Tag ::=
//       | ERA ::= an eraser
//       | NUM ::= numeric literal
//       | REF ::= a reference
//       | VAR ::= a variable
//       | CON ::= a constructor
//       | DUP ::= a duplicator
//       | OPR ::= numeric binary op
//       | SWI ::= numeric switch
//
// ## Memory Layout Example
//
// Consider, again, the following program:
//
//     (a b)
//     & (b a) ~ (x (y *))
//     & {y x} ~ @foo
//
// In memory, it could be represented as, for example:
//
// - RBAG | FST-TREE | SND-TREE
// - ---- | -------- | --------
// - 0800 | CON 0001 | CON 0002 // '& (b a) ~ (x (y *))'
// - 1800 | DUP 0005 | REF 0000 // '& {x y} ~ @foo'
// - ---- | -------- | --------
// - NODE | PORT-1   | PORT-2
// - ---- | -------- | --------
// - 0000 | CON 0001 |          // points to root node
// - 0001 | VAR 0000 | VAR 0001 // '(a b)' node (root)
// - 0002 | VAR 0001 | VAR 0000 // '(b a)' node
// - 0003 | VAR 0002 | CON 0004 // '(x (y *))' node
// - 0004 | VAR 0003 | DUP 0000 // '(y *)' node
// - 0005 | VAR 0003 | VAR 0002 // '{y x}' node
// - ---- | -------- | --------
//
// Notes:
// - There are 2 RBAG entries, to represent each redexes.
// - There are 5 NODE entries, to represent each CON/DUP node.
// - We store the root on PORT-1 of NODE 0000.
// - The RBAG buffer is partitioned:
//   - The fst redex is a CON-CON, stored on partition 0x0800.
//   - The snd redex is a DUP-CON, stored on partition 0x1800.
// - VARs always come in pairs. Notice that each VAR appears twice.

#include <cstdio>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

extern "C" {
#include "common.h"
}

// Port: Constructor and Getters
// -----------------------------

__device__ __host__ inline Port new_port(Tag tag, Val val) {
  return (val << TAG_BITS) | tag;
}

__device__ __host__ inline Port new_var(Val val) { return new_port(VAR, val); }

__device__ __host__ inline Port new_ref(Val val) { return new_port(REF, val); }

__device__ __host__ inline Port new_num(Val val) { return new_port(NUM, val); }

__device__ __host__ inline Port new_era() { return new_port(ERA, 0); }

__device__ __host__ inline Port new_con(Val val) { return new_port(CON, val); }

__device__ __host__ inline Port new_dup(Val val) { return new_port(DUP, val); }

__device__ __host__ inline Port new_opr(Val val) { return new_port(OPR, val); }

__device__ __host__ inline Port new_swi(Val val) { return new_port(SWI, val); }

__device__ __host__ inline Port new_nil() { return 0; }

__device__ __host__ inline Tag get_tag(Port port) { return port & TAG_MASK; }

__device__ __host__ inline Val get_val(Port port) { return port >> TAG_BITS; }

// Pair: Constructor and Getters
// -----------------------------

__device__ __host__ inline Pair new_pair(Port fst, Port snd) {
  return ((u32)fst << 16) | snd;
}

__device__ __host__ inline Port get_fst(Pair pair) { return pair >> 16; }

__device__ __host__ inline Port get_snd(Pair pair) { return pair & 0xFFFF; }

// Debug Printing
// --------------

struct Show {
  char x[12];
};

__device__ __host__ void put_u16(char *B, u16 val) {
  for (int i = 0; i < 4; i++, val >>= 4) {
    B[4 - i - 1] = "0123456789ABCDEF"[val & 0xF];
  }
}

__device__ __host__ Show show_port(Port port) {
  // NOTE: this is done like that because sprintf seems not to be working
  Show s;
  switch (get_tag(port)) {
  case VAR:
    memcpy(s.x, "VAR:", 4);
    put_u16(s.x + 4, get_val(port));
    break;
  case REF:
    memcpy(s.x, "REF:", 4);
    put_u16(s.x + 4, get_val(port));
    break;
  case ERA:
    memcpy(s.x, "ERA:____", 8);
    break;
  case NUM:
    memcpy(s.x, "NUM:", 4);
    put_u16(s.x + 4, get_val(port));
    break;
  case CON:
    memcpy(s.x, "CON:", 4);
    put_u16(s.x + 4, get_val(port));
    break;
  case DUP:
    memcpy(s.x, "DUP:", 4);
    put_u16(s.x + 4, get_val(port));
    break;
  case OPR:
    memcpy(s.x, "OPR:", 4);
    put_u16(s.x + 4, get_val(port));
    break;
  case SWI:
    memcpy(s.x, "SWI:", 4);
    put_u16(s.x + 4, get_val(port));
    break;
  }
  s.x[8] = '\0';
  return s;
}

__device__ Show show_rule(Rule rule) {
  Show s;
  switch (rule) {
  case LINK:
    memcpy(s.x, "LINK", 4);
    break;
  case VOID:
    memcpy(s.x, "VOID", 4);
    break;
  case ERAS:
    memcpy(s.x, "ERAS", 4);
    break;
  case ANNI:
    memcpy(s.x, "ANNI", 4);
    break;
  case COMM:
    memcpy(s.x, "COMM", 4);
    break;
  case OPER:
    memcpy(s.x, "OPER", 4);
    break;
  case SWIT:
    memcpy(s.x, "SWIT", 4);
    break;
  case CALL:
    memcpy(s.x, "CALL", 4);
    break;
  default:
    memcpy(s.x, "????", 4);
    break;
  }
  s.x[4] = '\0';
  return s;
}

__device__ __host__ void print_program(const Program *prog) {
  printf("RBAG | FST-TREE | SND-TREE\n");
  printf("---- | -------- | --------\n");
  for (u32 i = 0; i < RBAG_LEN; ++i) {
    Pair rdex = prog->RBAG_BUF[i];
    if (rdex != 0) {
      printf("%04X | %s | %s\n", i, show_port(get_fst(rdex)).x,
             show_port(get_snd(rdex)).x);
    }
  }
  printf("---- | -------- | --------\n");
  printf("NODE | PORT-1   | PORT-2  \n");
  printf("---- | -------- | --------\n");
  for (u32 i = 0; i < NODE_LEN; ++i) {
    Pair node = prog->NODE_BUF[i];
    if (node != 0) {
      printf("%04X | %s | %s\n", i, show_port(get_fst(node)).x,
             show_port(get_snd(node)).x);
    }
  }
  printf("---- | -------- |\n");
  printf("VARS | VALUE    |\n");
  printf("---- | -------- |\n");
  for (u32 i = 0; i < VARS_LEN; ++i) {
    if (prog->VARS_BUF[i] != 0) {
      printf("%04X | %s |\n", i, show_port(prog->VARS_BUF[i]).x);
    }
  }
}

// Utils
// -----

// Swaps two ports.
__device__ __host__ inline void swap(Port *a, Port *b) {
  Port x = *a;
  *a = *b;
  *b = x;
}

// TODO: OPTIMIZE THIS
__device__ __host__ inline bool has_loc(Port a) {
  return get_tag(a) >= ERA && get_val(a) > 0;
}

// TODO: OPTIMIZE THIS
__device__ __host__ inline bool has_var(Port a) { return get_tag(a) == VAR; }

// TODO: OPTIMIZE THIS
__device__ __host__ inline bool is_principal(Tag a) {
  return get_tag(a) != VAR;
}

// Given two tags, gets their interaction rule. Uses a u64mask lookup table.
__device__ __host__ inline Rule get_rule(Tag a, Tag b) {
  const u64 x =
      0b1000001001001010001000100001001001001110000011101111111000000000;
  const u64 y =
      0b1000111001000110001011100001111010110000111100001111000000000000;
  const u64 z =
      0b0111101010111010110100101110001011000000000000001111000000000000;
  const u64 i = ((u64)a << 3) | (u64)b;
  return (Rule)((x >> i & 1) | (y >> i & 1) << 1 | (z >> i & 1) << 2);
}

// Gets a rule's priority
__device__ __host__ inline bool is_high_priority(Rule rule) {
  return rule == LINK || rule == VOID || rule == ERAS || rule == ANNI;
}

// RBag
// ----

__device__ RBag rbag_new() {
  RBag rbag;
  rbag.hi_idx = 0;
  rbag.lo_idx = 31;
  return rbag;
}

__device__ void rbag_push(RBag *rbag, Pair redex) {
  Rule rule = get_rule(get_tag(get_fst(redex)), get_tag(get_snd(redex)));
  // printf("PUSH :: RULE %d | %d %d | %s - PRIORITY %d\n", rule,
  // get_tag(get_fst(redex)), get_tag(get_snd(redex)), show_rule(rule).x,
  // is_high_priority(rule));
  if (is_high_priority(rule)) {
    rbag->buf[rbag->hi_idx++] = redex;
  } else {
    rbag->buf[rbag->lo_idx--] = redex;
  }
}

__device__ Pair rbag_pop(RBag *rbag) {
  if (rbag->hi_idx > 0) {
    return rbag->buf[--rbag->hi_idx];
  } else if (rbag->lo_idx < 31) {
    return rbag->buf[++rbag->lo_idx];
  } else {
    return NONE;
  }
}

// Linking
// -------

// Links `A ~ B`
__device__ void link(Program *program, RBag *rbag, Port A, Port B) {
  u32 tid = threadIdx.x;

  // Attempts to directionally point `A ~> B`
  while (true) {
    // If `A` is PRI: swap `A` and `B`, and continue
    if (is_principal(get_tag(A))) {
      Port X = A;
      A = B;
      B = X;
    }

    // printf("[%04x] LINK %s ~ %s\n", tid, show_port(A).x, show_port(B).x);

    // If `A` is PRI: create the `A ~ B` redex
    if (is_principal(get_tag(A))) {
      // printf("[%04x] >> redex\n", tid);
      rbag_push(rbag, new_pair(A, B));
      break;
    }

    // While `B` is VAR: extend it (as an optimization)
    while (get_tag(B) == VAR) {
      // printf("[%04x] >> extend\n", tid);
      //  Takes the current `B` substitution as `B'`
      Port B_ = atomicExch(&program->VARS_BUF[get_val(B)], B);
      // If there was no `B'`, stop, as there is no extension
      if (B_ == B) {
        break;
      }
      // Otherwise, delete `B` (we own both) and continue as `A ~> B'`
      atomicExch(&program->VARS_BUF[get_val(B)], 0);
      B = B_;
    }

    // Since `A` is VAR: point `A ~> B`.
    if (true) {
      // printf("[%04x] >> point\n", tid);
      //  Stores `A -> B`, taking the current `A` subst as `A'`
      Port A_ = atomicExch(&program->VARS_BUF[get_val(A)], B);
      // If there was no `A'`, stop, as we lost ownership
      if (A_ == A) {
        break;
      }
      // Otherwise, delete `A` (we own both) and link `A' ~ B`
      atomicExch(&program->VARS_BUF[get_val(A)], 0);
      A = A_;
    }
  }
}

// Allocator
// ---------

// Finds the nth set bit in a word. Returns NONE if nth > bits_set.
__device__ u32 nth_set_bit(u32 nth, u32 len, u32 val) {
  if (__popc(val) <= nth) {
    return NONE;
  }
  u32 idx = 0;
  while (len > 0) {
    u32 mid = len >> 1;
    u32 rgt = val & ((1 << mid) - 1);
    u32 num = __popc(rgt);
    if (nth < num) {
      val = rgt;
    } else {
      nth -= num;
      idx += mid;
      val >>= mid;
    }
    len = mid;
  }
  return idx;
}

// The shared memory allocator operates in chunks of TPB consecutive reads, one
// per thread. Each thread checks if its slot is empty and broadcasts two masks:
// - `need`: bit N is set if thread N needs allocation
// - `free`: bit N is set if slot N is empty
// Threads needing allocation then compute their index based on their position
// in `need` and the `free` mask. This distributes free slots among threads
// until all allocations are satisfied or the array is fully scanned.
// Arguments:
// - out: output array to push allocated indices
// - num: number of indices to allocate
// - arr: array to find empty elements to allocate
// - len: length of the array
// Returns:
// - got: number of allocated elements
// template <typename T>
//__device__ T find(u32* out, u32 num, bool empty, T* arr, u32 len) {
// u32 tid = threadIdx.x;
// u32 LEN = blockDim.x;
// u32 wid = tid % 32;
// u32 got = 0;
// u32 loc = NONE;
// for (u32 ini = 0; ini < len; ini += LEN) {
////printf("[%04x] look %04x\n", tid, ini+tid);
// T elem = arr[ini+tid];
// u32 mask = __activemask();
// u32 need = __ballot_sync(mask, got < num);
// u32 good = __ballot_sync(mask, empty ? elem == 0 : elem > 0);
////printf("[%04x] mask=%08x need=%08x good=%08x\n", tid, mask, need, good);
// if (got < num) {
// u32 rank = __popc(need & ((1 << wid) - 1));
// u32 gotN = nth_set_bit(rank, 32, good);
// if (gotN != NONE) {
////printf("[%04x] got %04x\n", tid, ini + tid - wid + gotN);
// out[got++] = ini + tid - wid + gotN;
//}
//}
// u32 stop = __all_sync(mask, got >= num);
// if (stop) {
// break;
//}
//}
// return got;
//}

template <typename T> __device__ T alloc(u32 *out, u32 num, T *arr, u32 len) {
  u32 tid = threadIdx.x;
  u32 wid = tid % 32;
  u32 got = 0;
  for (u32 ini = 0; ini < len; ini += TPB) {
    // printf("[%04x] look %04x\n", tid, ini+tid);
    T elem = arr[ini + tid];
    if (elem == 0 && ini + tid > 0) {
      out[got++] = ini + tid;
    }
    if (got == num) {
      break;
    }
  }
  return got;
}

// Dereference
// -----------

__device__ inline Port adjust(Port port, u32 *NLOC, u32 *VLOC) {
  Tag tag = get_tag(port);
  Val val = get_val(port);
  if (has_loc(port)) {
    return new_port(tag, NLOC[val - 1]);
  }
  if (has_var(port)) {
    return new_port(tag, VLOC[val]);
  }
  return new_port(tag, val);
}

// Interaction Kernel
// ------------------

__constant__ Book D_BOOK;

__device__ u32 get_num(Program *program, Port N) {
  if (get_val(N) == 0) {
    return 0;
  } else {
    return atomicExch(&program->NODE_BUF[get_val(N)], 0);
  }
}

__global__ void interaction_kernel(Program *program_in, Program *program_out) {
  // INITIALIZATION
  // --------------

  // Shared Memory is a 96 KB Program.
  extern __shared__ char shared_mem[];
  Program *program = (Program *)shared_mem;

  // Thread Index
  const u32 tid = threadIdx.x;

  // Reads input program into shared memory
  // NOTE: does it in a way that avoids bank conflicts
  // NOTE: there are 256 Threads Per Block (TPB)
  for (u32 i = tid; i < RBAG_LEN; i += TPB) {
    program->RBAG_BUF[i] = program_in->RBAG_BUF[i];
  }
  for (u32 i = tid; i < NODE_LEN; i += TPB) {
    program->NODE_BUF[i] = program_in->NODE_BUF[i];
  }
  for (u32 i = tid; i < VARS_LEN; i += TPB) {
    program->VARS_BUF[i] = program_in->VARS_BUF[i];
  }

  __syncthreads();

  // Local redex bag.
  RBag rbag = rbag_new();

  // Allocations
  u32 NLOC[32]; // node locs
  u32 VLOC[32]; // vars locs

  //// Steals an initial redex.
  // Pair* got_rdx = &program->RBAG_BUF[tid];
  // if (*got_rdx != 0) {
  // rbag_push(&rbag, *got_rdx);
  // printf("[%04x] STEAL %s ~ %s\n", tid, show_port(get_fst(*got_rdx)).x,
  // show_port(get_snd(*got_rdx)).x); *got_rdx = 0;
  //}

  // Creates the initial redex.
  if (tid < TPB) {
    u32 got_node = alloc(NLOC, 1, program->VARS_BUF, VARS_LEN);
    program->NODE_BUF[NLOC[0]] = new_pair(new_era(), new_era());
    rbag_push(&rbag, new_pair(new_ref(2), new_con(NLOC[0])));
  }

  // Reduction loop.
  u32 rewrites = 0;
  while (true) {
    // Pops a redex.
    Pair r = rbag_pop(&rbag);

    // If there is no redex, break.
    if (r == NONE) {
      // printf("[%04x] DONE!\n", tid);
      break;
    }

    // Increments the rewrite rewrites.
    rewrites += 1;

    // Gets redex ports A and B.
    Port a = get_fst(r);
    Port b = get_snd(r);

    // Gets the rule type.
    Rule rule = get_rule(get_tag(a), get_tag(b));

    // debug-prints the rule type
    // printf("[%04x] RDEX %s ~ %s | RULE = %s | IDX = (%d,%d)\n", tid,
    // show_port(a).x, show_port(b).x, show_rule(rule).x, rbag.hi_idx,
    // rbag.lo_idx);

    // Interaction Table.
    // TODO: merge branches to reduce warp divergenze
    switch (rule) {

    case LINK: {
      // Swaps if needed
      if (get_tag(a) != VAR)
        swap(&a, &b);

      // Links
      link(program, &rbag, a, b);

      break;
    }

    case VOID: {
      break;
    }

    case ERAS: {
      // Swaps if needed
      if (get_tag(a) != ERA)
        swap(&a, &b);

      // Takes redex nodes
      Pair A = atomicExch(&program->NODE_BUF[get_val(a)], 0);

      // Gets ports
      Port B = b;
      Port A1 = get_fst(A);
      Port A2 = get_snd(A);

      // Links
      link(program, &rbag, A1, B);
      link(program, &rbag, A2, B);

      break;
    }

    case ANNI: {
      // Takes redex nodes
      Pair A = atomicExch(&program->NODE_BUF[get_val(a)], 0);
      Pair B = atomicExch(&program->NODE_BUF[get_val(b)], 0);

      // Gets ports
      Port A1 = get_fst(A);
      Port A2 = get_snd(A);
      Port B1 = get_fst(B);
      Port B2 = get_snd(B);

      // Links
      link(program, &rbag, A1, B1);
      link(program, &rbag, A2, B2);

      break;
    }

    case COMM: {
      // Takes redex nodes
      Pair A = atomicExch(&program->NODE_BUF[get_val(a)], 0);
      Pair B = atomicExch(&program->NODE_BUF[get_val(b)], 0);

      // Gets ports
      Port A1 = get_fst(A);
      Port A2 = get_snd(A);
      Port B1 = get_fst(B);
      Port B2 = get_snd(B);

      // Allocates nodes
      u32 got_nodes = alloc(NLOC, 4, program->NODE_BUF, NODE_LEN);
      if (got_nodes < 4) {
        // printf("[%04x] ERROR ALLOCATING NODES\n", tid); // TODO
        rewrites = 1 << 15;
        break;
      }

      // Allocates vars
      u32 got_vars = alloc(VLOC, 4, program->VARS_BUF, VARS_LEN);
      if (got_vars < 4) {
        // printf("[%04x] ERROR ALLOCATING VARS\n", tid); // TODO
        rewrites = 1 << 15;
        break;
      }

      // Creates vars
      Port v0 = new_var(VLOC[0]);
      Port v1 = new_var(VLOC[1]);
      Port v2 = new_var(VLOC[2]);
      Port v3 = new_var(VLOC[3]);

      // Stores vars
      program->VARS_BUF[VLOC[0]] = v0;
      program->VARS_BUF[VLOC[1]] = v1;
      program->VARS_BUF[VLOC[2]] = v2;
      program->VARS_BUF[VLOC[3]] = v3;

      // Stores nodes
      program->NODE_BUF[NLOC[0]] = new_pair(v0, v1);
      program->NODE_BUF[NLOC[1]] = new_pair(v2, v3);
      program->NODE_BUF[NLOC[2]] = new_pair(v0, v2);
      program->NODE_BUF[NLOC[3]] = new_pair(v1, v3);

      // Links
      link(program, &rbag, A1, new_port(get_tag(b), NLOC[0]));
      link(program, &rbag, A2, new_port(get_tag(b), NLOC[1]));
      link(program, &rbag, B1, new_port(get_tag(a), NLOC[2]));
      link(program, &rbag, B2, new_port(get_tag(a), NLOC[3]));

      break;
    }

    case OPER: {
      // Swaps if needed
      if (get_tag(a) != OPR)
        swap(&a, &b);

      // Takes redex nodes
      Pair A = atomicExch(&program->NODE_BUF[get_val(a)], 0);

      // Gets ports
      Port B = b; // num
      Port A1 = get_fst(A);
      Port A2 = get_snd(A);

      // TODO

      // if (get_tag(A1) == NUM) {
      //// Gets operand values
      // u16 av = get_num(program, A1);
      // u16 bv = get_num(program, b);

      //// TODO: implement table of operations
      // u16 rv = (av + bv) % 0xFF;

      //// Links
      // link(program, &rbag, A2, new_num(rv));
      //} else {
      //// Allocates nodes
      // u32 got_nodes = alloc(NLOC, 1, program->NODE_BUF, NODE_LEN);
      // if (got_nodes < 1) {
      // printf("[%04x] ERROR ALLOCATING NODES\n", tid); // TODO
      // rewrites = 1 << 15; break;
      //}

      //// Stores node
      // program->NODE_BUF[NLOC[0]] = new_pair(B, A2);

      //// Links
      // link(program, &rbag, new_opr(NLOC[0]), A1);
      //}

      break;
    }

    case SWIT: {
      // Swaps if needed
      if (get_tag(a) != SWI)
        swap(&a, &b);

      // Takes redex nodes
      Pair A = atomicExch(&program->NODE_BUF[get_val(a)], 0);

      // Gets ports
      Port B = b; // num
      Port A1 = get_fst(A);
      Port A2 = get_snd(A);

      if (B == new_num(0)) {
        // printf("[%04x] ###### is-zero\n", tid);
        //  Allocates nodes
        u32 got_nodes = alloc(NLOC, 1, program->NODE_BUF, NODE_LEN);
        if (got_nodes < 1) {
          // printf("[%04x] ERROR ALLOCATING NODES\n", tid); // TODO
          rewrites = 1 << 15;
          break;
        }

        // Stores node
        program->NODE_BUF[NLOC[0]] = new_pair(A2, new_era());

        // Links
        link(program, &rbag, new_port(CON, NLOC[0]), A1);
      } else {
        // Value
        u32 pred = get_num(program, B) - 1;
        u32 need = pred > 0 ? 3 : 2;

        // printf("[%04x] ###### is-succ %d\n", tid, pred);

        // Allocates nodes
        u32 got_nodes = alloc(NLOC, need, program->NODE_BUF, NODE_LEN);
        if (got_nodes < need) {
          // printf("[%04x] ERROR ALLOCATING NODES\n", tid); // TODO
          rewrites = 1 << 15;
          break;
        }

        // Stores nodes
        program->NODE_BUF[NLOC[0]] =
            new_pair(new_era(), new_port(CON, NLOC[1]));
        program->NODE_BUF[NLOC[1]] =
            new_pair(new_num(pred > 0 ? NLOC[2] : 0), A2);
        if (pred > 0) {
          program->NODE_BUF[NLOC[2]] = pred;
        }

        // Links
        link(program, &rbag, new_port(CON, NLOC[0]), A1);
      }

      break;
    }

    case CALL: {
      // Swaps if needed
      if (get_tag(a) != REF)
        swap(&a, &b);

      Port A = a; // ref
      Port B = b; // any

      // printf("->->-> fid=%d\n", get_val(A));

      // Gets the definition
      Def *def = &D_BOOK.DEFS_BUF[get_val(A)];

      // printf("->->-> vars_len=%d\n", def->VARS_LEN);

      // Allocates nodes
      // printf("[%04x] alloc %d nodes | fid=%d\n", tid, def->NODE_LEN-1,
      // get_val(A));
      u32 got_nodes =
          alloc(NLOC, def->_NODE_LEN - 1, program->NODE_BUF, NODE_LEN);
      if (got_nodes < def->_NODE_LEN - 1) {
        // printf("[%04x] ERROR ALLOCATING NODES\n", tid); // TODO
        rewrites = 1 << 15;
        break;
      }

      // Allocates vars
      // printf("[%04x] alloc %d vars | fid=%d\n", tid, def->VARS_LEN,
      // get_val(A));
      u32 got_vars = alloc(VLOC, def->_VARS_LEN, program->VARS_BUF, VARS_LEN);
      if (got_vars < def->_VARS_LEN) {
        // printf("[%04x] ERROR ALLOCATING VARS\n", tid); // TODO
        rewrites = 1 << 15;
        break;
      }

      // Stores vars
      for (u32 i = 0; i < def->_VARS_LEN; ++i) {
        program->VARS_BUF[VLOC[i]] = new_var(VLOC[i]);
      }

      // Stores nodes
      for (u32 i = 1; i < def->_NODE_LEN; ++i) {
        Pair nd = def->NODE_BUF[i];
        if ((def->MASK >> i) & 1) {
          // printf("[04x] new-number %04x\n", tid, nd);
          program->NODE_BUF[NLOC[i - 1]] = nd;
        } else {
          Port p1 = adjust(get_fst(nd), NLOC, VLOC);
          Port p2 = adjust(get_snd(nd), NLOC, VLOC);
          program->NODE_BUF[NLOC[i - 1]] = new_pair(p1, p2);
        }
      }

      // Writes each redex
      for (u32 i = 0; i < def->_RBAG_LEN; ++i) {
        Pair r = def->RBAG_BUF[i];
        Port a = adjust(get_fst(r), NLOC, VLOC);
        Port b = adjust(get_snd(r), NLOC, VLOC);
        link(program, &rbag, a, b);
      }

      // Returns the root
      Port root = adjust(get_fst(def->NODE_BUF[0]), NLOC, VLOC);
      // printf("[%04x] >> root | lnk %s ~ %s\n", tid, show_port(root).x,
      // show_port(B).x);
      link(program, &rbag, B, root);

      break;
    }
    }
  }

  __syncthreads();

  // printf("[%04x] %d rewrites\n", tid, rewrites);

  // RESULT
  // ------

  // Writes shared memory into output program
  for (u32 i = tid; i < RBAG_LEN; i += TPB) {
    program_out->RBAG_BUF[i] = program->RBAG_BUF[i];
  }
  for (u32 i = tid; i < NODE_LEN; i += TPB) {
    program_out->NODE_BUF[i] = program->NODE_BUF[i];
  }
  for (u32 i = tid; i < VARS_LEN; i += TPB) {
    program_out->VARS_BUF[i] = program->VARS_BUF[i];
  }

  atomicAdd(&program_out->rewrites, rewrites);
}

// Example Books
// -------------

//@loop    = (?<(#0 @loop$C0) a> a)
//@loop$C0 = (a b) & @loop ~ (a b)
//@main    = a & @loop ~ (#20 a)
const Book BOOK = {
    3,
    {
        {
            // loop
            0x00000000,
            0,
            {
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            },
            4,
            {
                0x000C0000, 0x001F0000, 0x00030009, 0x00140000, 0, 0, 0, 0,
                0,          0,          0,          0,          0, 0, 0, 0,
                0,          0,          0,          0,          0, 0, 0, 0,
                0,          0,          0,          0,          0, 0, 0, 0,
            },
            1,
        },
        {
            // loop$C0
            0x00000000,
            1,
            {
                0x00010014, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0,          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            },
            3,
            {
                0x000C0000, 0x00000008, 0x00000008, 0, 0, 0, 0, 0, 0, 0, 0,
                0,          0,          0,          0, 0, 0, 0, 0, 0, 0, 0,
                0,          0,          0,          0, 0, 0, 0, 0, 0, 0,
            },
            2,
        },
        {
            // main
            0x00000004,
            1,
            {
                0x0001000C, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0,          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            },
            3,
            {
                0x00000000, 0x00130000, 0x0008FFFF, 0, 0, 0, 0, 0, 0, 0, 0,
                0,          0,          0,          0, 0, 0, 0, 0, 0, 0, 0,
                0,          0,          0,          0, 0, 0, 0, 0, 0, 0,
            },
            1,
        },
    }};

// Main
// ----
extern "C" int _main() {
  return 0;
  Program prog_in;
  memset(&prog_in, 0, sizeof(Program));
  // boot(BOOK, 2, &prog_in, 2);
  print_program(&prog_in);

  // Copy the Book to the constant memory before launching the kernel
  cudaMemcpyToSymbol(D_BOOK, &BOOK, sizeof(Book));

  Program *d_prog_in;
  Program *d_prog_out;
  cudaMalloc(&d_prog_in, sizeof(Program));
  cudaMalloc(&d_prog_out, sizeof(Program));
  cudaMemcpy(d_prog_in, &prog_in, sizeof(Program), cudaMemcpyHostToDevice);

  cudaFuncSetAttribute(interaction_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sizeof(Program));
  interaction_kernel<<<SMS, TPB, sizeof(Program)>>>(d_prog_in, d_prog_out);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch interaction_kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  Program prog_out;
  cudaMemcpy(&prog_out, d_prog_out, sizeof(Program), cudaMemcpyDeviceToHost);

  printf("rewrites: %llu\n", prog_out.rewrites);
  // print_program(&prog_out);

  // printf("hi\n");

  return 0;
}
