// INFO: https://twitter.com/VictorTaelin/status/1772389464628670705
// Modified from
// https://gist.github.com/VictorTaelin/a92498156d49ae79ea55eee65944a1b4

// ./cuda_info.txt//

// HVM-CUDA: an Interaction Combinator evaluator in CUDA.
//
// # Format
//
// An HVM net is a graph with 4 node types:
// - *     ::= ERAser node. Has 0 aux ports.
// - (a b) ::= CONstructor node. Has 2 aux ports.
// - {a b} ::= DUPlicator node. Has 2 aux port.
// - @def  ::= lazy reference (to a global definition).
//
// Nodes form a tree-like structure in memory. For example:
//
//     ((* x) {x (y y)})
//
// Is interpreted as:
//
//     (CON (CON ERA x) (DUP x (CON y y)))
//
// That is, it is a tree with 5 nodes. The lowcase letters are variables. They
// represent 2 ends of a wire. So, for example, in the tree above, the first 'x'
// is connected to the second 'x'. Variables are always paired.
//
// A program consists of a root tree, plus list of active pairs. Example:
//
//     (a b)
//     & (b a) ~ (x (y *))
//     & {y x} ~ @foo
//
// The program above has 1 root 2 active trees pair <tree_0> ~ <tree_1>`). An
// active pair is two trees that are connected, and can interact. See below.
//
// # Interactions
//
// Active pairs, also called redexes, are reduced via an *interaction rule*.
// There are 6 rules, one for each possible combination of node types:
//
//     & * ~ * =>
//       (nothing)
//
//     & (a0 a1) ~ * =>
//       & a0 ~ *
//       & a1 ~ *
//
//     & {a0 a1} ~ * =>
//       & a0 ~ *
//       & a1 ~ *
//
//     & (a0 a1) ~ (b0 b1) =>
//       a0 ~ b0
//       a1 ~ b1
//
//     & {a0 a1} ~ {b0 b1} =>
//       a0 ~ b0
//       a1 ~ b1
//
//     & (a0 a1) ~ {b0 b1} =>
//       & a0 ~ {x y}
//       & a1 ~ {z w}
//       & b0 ~ (x z)
//       & b1 ~ (y w)
//
// Note:
// - ERA-ERA returns nothing (it is erased).
// - ERA-CON and ERA-DUP erase the CON/DUP and propagates the ERA.
// - CON-CON and DUP-DUP erase both nodes and link neighbors.
// - CON-DUP duplicates both notes, commuting internal wires.
//
// The process of applying these rules repeatedly is Turing Complete.
//
// # Definitions
//
// A top-level definition is just a statically known closed net, also called a
// package. It is represented like a program, with a root and linked trees:
//
//     @foo = (a b)
//       & @tic ~ (x a)
//       & @tac ~ (x b)
//
// The statement above represents a definition, @foo, with the `(a b)` tree as
// the root, and two linked trees: `@tic ~ (x a)` and `@tac ~ (x b)`.
//
// Definitions interact by unrolling, or expanding, to their values, whenever
// they're part of an active pair. For example:
//
//     ...
//     & @foo ~ ((* a) (* a))
//     ...
//
// The program above would expand @foo, resulting in:
//
//     ...
//     & (a0 b0) ~ ((* a) (* a))
//     & @tic ~ (x0 a0)
//     & @tac ~ (x0 b0)
//     ...
//
// Note that:
// 1. We replaced the @foo node by the root of the @foo definition.
// 2. We added each redex of @foo (`@tic ~ ...` and `@tac ~ ...`) to the
// program.
// 3. To avoid name capture, we appended `0` to `x`, `a`, and `b`.
// This program would then proceed to reduce the remaining redexes.
//
// Also, as an optimization, whenever a @def interacts with a node not contained
// in that definition, it will just copy the @def. For example, consider:
//
//     & @foo ~ {a b}
//
// Normally, the redex above would expand @foo, but there are no DUP nodes in
// the @foo's definition. As such, it will, instead, just copy @foo:
//
//     a ~ @foo
//     b ~ @foo
//
// Also, as a similar optimization, `@foo ~ ERA` will just erase the ref node.
// That's correct, because references are closed terms, which are always fully
// collected by erasure interactions, so, there is no need to unroll it.
//
// # Example Reduction
//
// Consider the original program:
//
//     (a b)
//     & (b a) ~ (x (y *))
//     & {y x} ~ @foo
//
// It has two redexes. HVM is strongly confluent, which means we can reduce
// redexes in any order (and even in parallel) without affecting the total
// interaction count. Let's start reducing the first active pair. It is a
// CON-CON interaction, which reduces to:
//
//     (a b)
//     & b ~ x
//     & a ~ (y *)
//     & {y x} ~ @foo
//
// Now, we must copy @foo, since it doesn't contain DUP nodes:
//
//     (a b)
//     & b ~ x
//     & a ~ (y *)
//     & y ~ @foo
//     & x ~ @foo
//
// Now, we just resolve the 'y ~ @foo' and 'x ~ @foo' wirings:
//
//     (a b)
//     & b ~ @foo
//     & a ~ (@foo *)
//
// Finally, we resolve the 'b ~ (@foo *)' and 'a ~ @foo' wirings:
//
//     ((@foo *) @foo)
//
// This program is in un-expanded normal form, which can still contain
// references (because they never were expanded by the usual rules). We
// must proceed to expand them and reduce further:
//
//     ((a0 b0) (a1 b1))
//       & @tic ~ (x0 a0)
//       & @tac ~ (x0 b0)
//       & @tic ~ (x1 a1)
//       & @tac ~ (x1 b1)
//
// At this point, we must expand @tic and @tac. Suppose they were just:
//
//     @tic = (k k)
//     @tac = (k k)
//
// This would expand to:
//
//     ((a0 b0) (a1 b1))
//       & (k0 k0) ~ (x0 a0)
//       & (k1 k1) ~ (x0 b0)
//       & (k2 k2) ~ (x1 a1)
//       & (k3 k3) ~ (x1 b1)
//
// Let's now perform all these CON-CON interactions in parallel:
//
//     ((a0 b0) (a1 b1))
//       & k0 ~ x0
//       & k0 ~ a0
//       & k1 ~ x0
//       & k1 ~ b0
//       & k2 ~ x1
//       & k2 ~ a1
//       & k3 ~ x1
//       & k3 ~ b1
//
// Let's now resolve the kN wires:
//
//     ((a0 b0) (a1 b1))
//       & x0 ~ a0
//       & x0 ~ b0
//       & x1 ~ a1
//       & x1 ~ b1
//
// Let's now resolve the xN wires:
//
//     ((a0 b0) (a1 b1))
//       & a0 ~ b0
//       & a1 ~ b1
//
// Let's now resolve the aN wires:
//
//     ((b0 b0) (b1 b1))
//
// This is the final result. The program is now in normal form.
//
// # Memory Layout
//
// HVM-CUDA represents programs as tuple:
//
//     Program ::= {
//       RBAG : [Pair] // pairs (redex bag)
//       NODE : [Pair] // pairs (node buffer)
//       SUBS : [Port] // ports (substitutions)
//     }
//
// A definition is like a program, with varlen buffers:
//
//     Def ::= {
//       PAIR : [Pair]
//       NODE : [Node]
//     }
//
// A Book is a buffer of definitions:
//
//   Book ::= {
//     DEFS: [Def]
//   }
//
// A Pair consists of two Ports, representing a either a redex or a node:
//
//     Pair ::= (Port, Port)
//
// A Port consists of a tag and a value:
//
//     Port ::= 3-bit tag + 14-bit val
//
// There are 9 Tags:
//
//     Tag ::=
//       | ERA ::= an erasure ndoe
//       | CON ::= the CON node
//       | DUP ::= the DUP node
//       | VAR ::= a variable
//       | REF ::= a reference
//       | NUM ::= numeric literal
//       | OP2 ::= numeric binary op
//       | OP1 ::= numeric unary op
//       | SWI ::= numeric switch
//
// Since 2 bits can only represent 4 tags, we represent ERA as DUP(0), and leave
// CON(0) for empty spaces. That means the first index of the NODE buffer can't
// be addressed, so we use it to store the root port.
//
// The RBAG buffer is split into 4 chunks of 1536 elems, 1 for each interaction:
// - Collection   : ERA-CON and ERA-DUP interactions - indices 0x0000 to 0x05FF
// - Annihilation : CON-CON and DUP-DUP interactions - indices 0x0600 to 0x0BFF
// - Commutation  : CON-DUP interaction              - indices 0x1200 to 0x17FF
// - Dereference  : REF interactions                 - indices 0x1800 to 0x1DFF
// This allows threads to perform the same interaction, reducing divergence.
// Note that the ERA-ERA rule isn't included, because it is skipped enirely.
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
//
// # The Interaction Kernel
//
// The heart of HVM's reduction engine is the interaction kernel, which is
// executed by 256 to 1024 threads in parallel in a single SM (multprocessor),
// depending on the GPU configuration. These are grouped in "squads" of 4
// threads each. So, for example, if we launch 256 threads per block, then we'll
// have 128 squads. This is done because interaction rules can be further be
// broken down in 4 sequential fragments. This increases the maximum theoretical
// speedup by 4x, per Amdahl's law!
//
// The interaction kernel works as follows:

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

// Integers
// --------

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef unsigned long long int u64;

// Configuration
// -------------

const u32 TPB = 256; // Threads Per Block

// Types
// -----

typedef u8 Tag;   // tag: 3-bit (rounded up to u8)
typedef u16 Val;  // val: 13-bit (rounded up to u16)
typedef u16 Port; // port: tag + val (fits an u16 exactly)
typedef u32 Pair; // pair: port + port

// Tags
const Tag CON = 0; // CON node
const Tag DUP = 1; // DUP node
const Tag VAR = 2; // variable
const Tag REF = 3; // reference
const Tag NUM = 4;
const Tag OP2 = 5;
const Tag OP1 = 6;
const Tag SWI = 7;
const Tag ERA = 8;  // erasure
const Tag NIL = 9;  // empty space
const Tag SUB = 10; // incoming substitution

// Port Number
const Tag P1 = 0; // PORT-1
const Tag P2 = 1; // PORT-2

// Rule Type
const Tag ANNI = 0; // CON-CON or DUP-DUP
const Tag COMM = 1; // CON-DUP
const Tag ERAS = 2; // ERA-CON or ERA-DUP
const Tag CALL = 3; // REF-ANY

// Program
const u32 RBAG_LEN = 0x1000; // max 4096 redexes
const u32 NODE_LEN = 0x2000; // max 8196 nodes
const u32 SUBS_LEN = 0x2000; // max 8196 subs
struct Program {
  u32 RBAG_BUF[RBAG_LEN];
  u32 NODE_BUF[NODE_LEN];
  u32 SUBS_BUF[SUBS_LEN]; // NOTE: can't use u16 due to no atomicExch
};

// Definition
struct Def {
  u32 VARS_LEN;
  u32 RBAG_LEN;
  Pair *RBAG_BUF;
  u32 NODE_LEN;
  Pair *NODE_BUF;
};

// Book
struct Book {
  u32 DEFS_LEN;
  Def *DEFS_BUF;
};

// Local Redex Bag
struct RBag {
  u32 len;
  u32 buf[32];
};

// None
const u32 NONE = 0xFFFFFFFF;

// Port: Constructor and Getters
// -----------------------------

__device__ __host__ inline Port new_port(Tag tag, Val val) {
  return (val << 3) | tag;
}

__device__ __host__ inline Port new_con(u32 val) { return new_port(CON, val); }

__device__ __host__ inline Port new_dup(u32 val) { return new_port(DUP, val); }

__device__ __host__ inline Port new_var(u32 val) { return new_port(VAR, val); }

__device__ __host__ inline Port new_ref(u32 val) { return new_port(REF, val); }

__device__ __host__ inline Port new_nil() { return new_port(CON, 0); }

__device__ __host__ inline Port new_era() { return new_port(DUP, 0); }

__device__ __host__ inline Port new_sub() { return 0xFFFF; }

__device__ __host__ inline Tag get_tag(Port port) {
  return port == new_nil()   ? NIL
         : port == new_era() ? ERA
         : port == new_sub() ? SUB
                             : port & 7;
}

__device__ __host__ inline Val get_val(Port port) { return port >> 3; }

// TODO: get_rule: given 2 tags, returns the rule type
__device__ __host__ inline Tag get_rule(Tag fst_tag, Tag snd_tag) {
  return fst_tag == ERA       ? ERAS
         : snd_tag == ERA     ? ERAS
         : fst_tag == REF     ? CALL
         : snd_tag == REF     ? CALL
         : fst_tag == snd_tag ? ANNI
                              : COMM;
}

// Pair: Constructor and Getters
// -----------------------------

__device__ __host__ inline Pair new_pair(Port fst, Port snd) {
  return ((u32)fst << 16) | snd;
}

__device__ __host__ inline Port get_fst(Pair pair) { return pair >> 16; }

__device__ __host__ inline Port get_snd(Pair pair) { return pair & 0xFFFF; }

// Program
// -------

// Loads a book definition into an empty program
void boot(const Book book, u32 fid, Program *prog) {
  // Zeroes the program
  memset(prog, 0, sizeof(Program));

  // Find definition
  Def def = book.DEFS_BUF[fid];

  // Copy into program
  const u32 pair_len = def.RBAG_LEN;
  const u32 node_len = def.NODE_LEN;
  const u32 vars_len = def.VARS_LEN;
  printf(">> %d %d %d\n", pair_len, node_len, vars_len);
  for (u32 i = 0; i < pair_len; ++i) {
    // printf("- %08x\n", def.RBAG_BUF[i]);
    prog->RBAG_BUF[i] = def.RBAG_BUF[i];
  }
  for (u32 i = 0; i < node_len; ++i) {
    prog->NODE_BUF[i] = def.NODE_BUF[i];
    // printf("- %08x\n", def.NODE_BUF[i]);
  }
  for (u32 i = 0; i < vars_len; ++i) {
    prog->SUBS_BUF[i] = new_sub();
    // printf("- %08x\n", def.NODE_BUF[i]);
  }
}

// Utils
// -----

__device__ __host__ inline Pair swap(Pair *ptr, Pair val) {
  Pair old = *ptr;
  *ptr = val;
  return old;
}

// Debug Printing
// --------------

struct ShowPort {
  char x[9];
};

__device__ __host__ void put_u16(char *B, u16 val) {
  for (int i = 0; i < 4; i++, val >>= 4) {
    B[4 - i - 1] = "0123456789ABCDEF"[val & 0xF];
  }
}

// NOTE: this is done like that because sprintf seems not to be working
__device__ __host__ ShowPort show_port(Port port) {
  ShowPort s;
  switch (get_tag(port)) {
  case NIL:
    memcpy(s.x, "___:____", 8);
    break;
  case ERA:
    memcpy(s.x, "ERA:____", 8);
    break;
  case SUB:
    memcpy(s.x, "SUB:____", 8);
    break;
  case CON:
    memcpy(s.x, "CON:", 4);
    put_u16(s.x + 4, get_val(port));
    break;
  case DUP:
    memcpy(s.x, "DUP:", 4);
    put_u16(s.x + 4, get_val(port));
    break;
  case VAR:
    memcpy(s.x, "VAR:", 4);
    put_u16(s.x + 4, get_val(port));
    break;
  case REF:
    memcpy(s.x, "REF:", 4);
    put_u16(s.x + 4, get_val(port));
    break;
  case NUM:
    memcpy(s.x, "NUM:", 4);
    put_u16(s.x + 4, get_val(port));
    break;
  case OP2:
    memcpy(s.x, "OP2:", 4);
    put_u16(s.x + 4, get_val(port));
    break;
  case OP1:
    memcpy(s.x, "OP1:", 4);
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
  printf("SUBS | VALUE    |\n");
  printf("---- | -------- |\n");
  for (u32 i = 0; i < SUBS_LEN; ++i) {
    if (prog->SUBS_BUF[i] != 0) {
      printf("%04X | %s |\n", i, show_port(prog->SUBS_BUF[i]).x);
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
template <typename T>
__device__ T find(u32 *out, u32 num, bool empty, T *arr, u32 len) {
  u32 tid = threadIdx.x;
  u32 LEN = blockDim.x;
  u32 wid = tid % 32;
  u32 got = 0;
  u32 loc = NONE;
  for (u32 ini = 0; ini < len; ini += LEN) {
    // printf("[%04x] look %04x\n", tid, ini+tid);
    T elem = arr[ini + tid];
    u32 mask = __activemask();
    u32 need = __ballot_sync(mask, got < num);
    u32 good = __ballot_sync(mask, empty ? elem == 0 : elem > 0);
    // printf("[%04x] mask=%08x need=%08x good=%08x\n", tid, mask, need, good);
    if (got < num) {
      u32 rank = __popc(need & ((1 << wid) - 1));
      u32 gidx = nth_set_bit(rank, 32, good);
      // printf("[%04x] rank=%08x gidx=%08x\n", tid, rank, gidx);
      if (gidx != NONE) {
        out[got++] = ini + gidx;
      }
    }
    u32 stop = __all_sync(mask, got >= num);
    if (stop) {
      break;
    }
  }
  return got;
}

template <typename T> __device__ T alloc(u32 *out, u32 num, T *arr, u32 len) {
  return find(out, num, true, arr, len);
}

template <typename T> __device__ T take(u32 *out, u32 num, T *arr, u32 len) {
  return find(out, num, false, arr, len);
}

// Example Books
// -------------

//@F    = ((@FS ((a a) b)) b)
//@FS   = ({a b} c) & @F ~ (a (d c)) & @F ~ (b d)
//@S    = (a ((a b) (* b)))
//@Z    = (* (a a))
//@main = a & @F ~ (b a) & @S ~ (c b) & @S ~ (@Z c)
// (TODO)

//@main = a & ({(b c) (d b)} (d c)) ~ ((e e) a)
// RBAG | FST-TREE | SND-TREE
//---- | -------- | --------
// 0000 | ___ ____ | CON 0005
//---- | -------- | --------
// NODE | PORT-1   | PORT-2
//---- | -------- | --------
// 0000 | DUP 0001 | CON 0004
// 0001 | CON 0002 | CON 0003
// 0002 | VAR 0001 | VAR 0002
// 0003 | VAR 0003 | VAR 0001
// 0004 | VAR 0003 | VAR 0002
// 0005 | CON 0006 | VAR 0000
// 0006 | VAR 0004 | VAR 0004
//---- | -------- |
// SUBS | VALUE    |
//---- | -------- |
// const Book BOOK = {
// 1,
//(Def[]){
//{ // main
// 5, // TODO: generate correct var count
// 1, (Pair[]){ 0x00080030, },
// 8, (Pair[]){ 0x00020000, 0x00110028, 0x00180020, 0x000A0012, 0x001A000A,
// 0x001A0012, 0x00380002, 0x00220022, },
//},
//}
//};

//@main = a & ({(b c) (d b)} (d c)) ~ (((e (f g)) (f (e g))) a)
// const Book BOOK = {
// 1,
//(Def[]){
//{ // main
// 7, // TODO: generate correct var count
// 1, (Pair[]){ 0x00080030, },
// 12, (Pair[]){ 0x00020000, 0x00110028, 0x00180020, 0x000A0012, 0x001A000A,
// 0x001A0012, 0x00380002, 0x00400050, 0x00220048, 0x002A0032, 0x002A0058,
// 0x00220032, },
//},
//}
//};

//@loop = (?<(#0 @loop$C0) a> a)

//@main = a
//& @loop ~ (#20 a)
const Book BOOK = {3, (Def[]){
                          {
                              // loop
                              7, // FIXME: generate correct var count
                              0,
                              (Pair[]){NULL},
                              4,
                              (Pair[]){
                                  0x00080000,
                                  0x001F0002,
                                  0x0004000B,
                                  0x00100002,
                              },
                          },
                          {
                              // loop$C0
                              7, // FIXME: generate correct var count
                              1,
                              (Pair[]){
                                  0x00030010,
                              },
                              3,
                              (Pair[]){
                                  0x00080000,
                                  0x0002000A,
                                  0x0002000A,
                              },
                          },
                          {
                              // main
                              7, // FIXME: generate correct var count
                              1,
                              (Pair[]){
                                  0x00030008,
                              },
                              2,
                              (Pair[]){
                                  0x00020000,
                                  0x00A40002,
                              },
                          },
                      }};

// loop
// 0000 | SWI:0002 | VAR:0000
// 0001 | NUM:0000 | SWI:0002
// 0002 | CON:0001 | VAR:0000

// main
// 0000 | REF:0000 | CON:0001
//---- | -------- | --------
// 0000 | VAR:0000 | ___:____
// 0001 | VAR:0000 | VAR:0000
// 0002 | ___:____ | ___:____

// Dereference
// -----------

__device__ inline Port adjust(Port port, u32 *node_locs, u32 *subs_locs) {
  Tag tag = get_tag(port);
  Val val = get_val(port);
  if (tag <= DUP) {
    // TODO: WHAT?
    return new_port(tag, node_locs[val - 1]);
  }
  if (tag == VAR) {
    return new_port(tag, subs_locs[val]);
  }
  return new_port(tag, val);
}

//__device__ Port deref(Book* book, Program* program, u32 fid) {
// u32 rbag_locs[32]; // allocated redexes
// u32 node_locs[32]; // allocated nodes
// u32 subs_locs[32]; // fresh vars

//// Gets the definition
// Def *def = &book->DEFS_BUF[fid];

//// Allocates redexes
// alloc(rbag_locs, def->RBAG_LEN, program->RBAG_BUF, RBAG_LEN);

//// Allocates nodes
// alloc(node_locs, def->NODE_LEN-1, program->NODE_BUF, NODE_LEN);

//// Allocates vars
// alloc(subs_locs, def->VARS_LEN, program->SUBS_BUF, SUBS_LEN);

//// Writes each node
// for (u32 i = 1; i < def->NODE_LEN; ++i) {
// Pair nd = def->NODE_BUF[i];
// Port p1 = adjust(get_fst(nd), node_locs, subs_locs);
// Port p2 = adjust(get_snd(nd), node_locs, subs_locs);
// program->NODE_BUF[node_locs[i-1]] = new_pair(p1, p2);
//}

//// Writes each redex
// for (u32 i = 0; i < def->RBAG_LEN; ++i) {
// Pair rx = def->RBAG_BUF[i];
// Port p1 = adjust(get_fst(rx), node_locs, subs_locs);
// Port p2 = adjust(get_snd(rx), node_locs, subs_locs);
// program->RBAG_BUF[rbag_locs[i]] = new_pair(p1, p2);
//}

//// Returns the root
// return adjust(def->NODE_BUF[0], node_locs, subs_locs);
//}

// Interaction Kernel
// ------------------

// Links `a ~ b`.
__device__ void link(Program *program, RBag *rbag, Port a, Port b) {
  u32 tid = threadIdx.x;

  while (true) {
    printf("[%04x] lnk %s ~ %s\n", tid, show_port(a).x, show_port(b).x);

    // If both are main ports, create a new redex.
    if (get_tag(a) <= DUP && get_tag(b) <= DUP) {
      // printf("[%04x] link: is redex\n", tid);
      rbag->buf[rbag->len++] = new_pair(a, b);
      break;
    }

    // printf("[%04x] link: is subst\n", tid, show_port(a).x, show_port(b).x);

    // Otherwise, check if one is a var `x`.
    Port *x = NULL;
    Port t;
    if (get_tag(a) == VAR) {
      x = &a;
      t = b;
    } else if (get_tag(b) == VAR) {
      x = &b;
      t = a;
    }

    // If yes, substitute `x <- other`.
    if (x != NULL) {
      u16 var = get_val(*x);
      printf("[%04x] set var %04x = %04x\n", tid, var, t);
      Port got = atomicExch(&program->SUBS_BUF[var], t);
      // printf("[%04x] link: subst %d <- %s\n", tid, var, show_port(t).x);
      printf("[%04x] got = %04x\n", tid, got);
      // If `x` had a value, clear it and link `value ~ other`.
      if (got == 0) {
        printf("[%04x] LINK ERROR - UNREACHABLE\n", tid);
      }
      if (got != new_sub()) {
        // printf("[%04x] REPEAT!\n", tid);
        printf("[%04x] set var %04x = %04x\n", var, 0);
        atomicExch(&program->SUBS_BUF[var], 0);
        *x = got;
      } else {
        break;
      }
    } else {
      break;
    }
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
  const u32 LEN = blockDim.x;

  // Reads input program into shared memory
  // NOTE: does it in a way that avoids bank conflicts
  // NOTE: there are 256 Threads Per Block (TPB)
  for (u32 i = tid; i < RBAG_LEN; i += LEN) {
    program->RBAG_BUF[i] = program_in->RBAG_BUF[i];
  }
  for (u32 i = tid; i < NODE_LEN; i += LEN) {
    program->NODE_BUF[i] = program_in->NODE_BUF[i];
  }
  for (u32 i = tid; i < SUBS_LEN; i += LEN) {
    program->SUBS_BUF[i] = program_in->SUBS_BUF[i];
  }

  __syncthreads();

  // Local redex bag.
  RBag rbag;
  rbag.len = 0;

  // Allocations
  u32 node_locs[32];
  u32 subs_locs[32];

  // Steals an initial redex.
  Pair *got_rdx = &program->RBAG_BUF[tid];
  if (*got_rdx != 0) {
    rbag.buf[rbag.len++] = *got_rdx;
    *got_rdx = 0;
  }

  // Reduction loop.
  while (true) {
    // If there is no redex, break.
    if (rbag.len == 0) {
      printf("[%04x] DONE!\n", tid);
      break;
    }

    // Pops a redex.
    Pair rdx = rbag.buf[--rbag.len];

    // printf("[%04x] REDUCING: %08x\n", tid, rdx);

    // debug-prints the rule type
    const Port a = get_fst(rdx);
    const Port b = get_snd(rdx);

    switch (get_rule(get_tag(a), get_tag(b))) {
    case ANNI: {
      printf("[%04x] ANNI %s | %s\n", tid, show_port(a).x, show_port(b).x);

      // Takes redex nodes
      Pair a_node = atomicExch(&program->NODE_BUF[get_val(a)], 0);
      Pair b_node = atomicExch(&program->NODE_BUF[get_val(b)], 0);
      // printf("[%04x] a_node = %s %s\n", tid, show_port(get_fst(a_node)).x,
      // show_port(get_snd(a_node)).x); printf("[%04x] b_node = %s %s\n", tid,
      // show_port(get_fst(b_node)).x, show_port(get_snd(b_node)).x);

      // Links neighbors
      link(program, &rbag, get_fst(a_node), get_fst(b_node));
      link(program, &rbag, get_snd(a_node), get_snd(b_node));

      break;
    }

    case COMM: {
      printf("[%04x] COMM %s | %s\n", tid, show_port(a).x, show_port(b).x);

      // Takes redex nodes
      Pair a_node = atomicExch(&program->NODE_BUF[get_val(a)], 0);
      Pair b_node = atomicExch(&program->NODE_BUF[get_val(b)], 0);
      // printf("[%04x] a_node = %s %s\n", tid, show_port(get_fst(a_node)).x,
      // show_port(get_snd(a_node)).x); printf("[%04x] b_node = %s %s\n", tid,
      // show_port(get_fst(b_node)).x, show_port(get_snd(b_node)).x);

      // Allocates new nodes
      u32 got_nodes = alloc(node_locs, 4, program->NODE_BUF, NODE_LEN);
      if (got_nodes < 4) {
        printf("[%04x] ERROR ALLOCATING NODES\n", tid); // TODO
      }

      // Allocates new vars
      u32 got_vars = alloc(subs_locs, 4, program->SUBS_BUF, SUBS_LEN);
      if (got_vars < 4) {
        printf("[%04x] ERROR ALLOCATING VARS\n", tid); // TODO
      }

      // TODO: print each elem in got_nodes and got_vars (for debugging)
      // for (int i = 0; i < 4; ++i) {
      // printf("[%04x] - node %04x\n", tid, node_locs[i]);
      // printf("[%04x] - var  %04x\n", tid, subs_locs[i]);
      //}

      // Gets fresh locs
      u16 l0 = node_locs[0];
      u16 l1 = node_locs[1];
      u16 l2 = node_locs[2];
      u16 l3 = node_locs[3];

      // Gets fresh vars
      u16 s0 = subs_locs[0];
      u16 s1 = subs_locs[1];
      u16 s2 = subs_locs[2];
      u16 s3 = subs_locs[3];

      printf(
          "[%04x] alloc nodes %04x %04x %04x %04x | vars %04x %04x %04x %04x\n",
          tid, l0, l1, l2, l3, s0, s1, s2, s3);

      // Creates new vars
      Port v0 = new_port(VAR, s0);
      Port v1 = new_port(VAR, s1);
      Port v2 = new_port(VAR, s2);
      Port v3 = new_port(VAR, s3);

      // Stores new vars
      program->SUBS_BUF[s0] = new_sub();
      program->SUBS_BUF[s1] = new_sub();
      program->SUBS_BUF[s2] = new_sub();
      program->SUBS_BUF[s3] = new_sub();

      // Stores new nodes
      program->NODE_BUF[l0] = new_pair(v0, v1);
      program->NODE_BUF[l1] = new_pair(v2, v3);
      program->NODE_BUF[l2] = new_pair(v0, v2);
      program->NODE_BUF[l3] = new_pair(v1, v3);

      // Links neighbors
      link(program, &rbag, new_port(get_tag(a), l0), get_fst(b_node));
      link(program, &rbag, new_port(get_tag(a), l1), get_snd(b_node));
      link(program, &rbag, new_port(get_tag(b), l2), get_fst(a_node));
      link(program, &rbag, new_port(get_tag(b), l3), get_snd(a_node));

      break;
    }

    case ERAS: {
      printf("[%04x] ERAS %s | %s\n", tid, show_port(a).x, show_port(b).x);
      break;
    }

    case CALL: {
      printf("[%04x] CALL %s | %s\n", tid, show_port(a).x, show_port(b).x);
      break;
    }
    }
  }

  __syncthreads();

  // RESULT
  // ------

  // Writes shared memory into output program
  for (u32 i = tid; i < RBAG_LEN; i += LEN) {
    program_out->RBAG_BUF[i] = program->RBAG_BUF[i];
  }
  for (u32 i = tid; i < NODE_LEN; i += LEN) {
    program_out->NODE_BUF[i] = program->NODE_BUF[i];
  }
  for (u32 i = tid; i < SUBS_LEN; i += LEN) {
    program_out->SUBS_BUF[i] = program->SUBS_BUF[i];
  }
}

int main() {
  Program prog_in;
  boot(BOOK, 0, &prog_in);
  print_program(&prog_in);

  Program *d_prog_in;
  Program *d_prog_out;
  cudaMalloc(&d_prog_in, sizeof(Program));
  cudaMalloc(&d_prog_out, sizeof(Program));
  cudaMemcpy(d_prog_in, &prog_in, sizeof(Program), cudaMemcpyHostToDevice);

  cudaFuncSetAttribute(interaction_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       sizeof(Program));
  interaction_kernel<<<1, TPB, sizeof(Program)>>>(d_prog_in, d_prog_out);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch interaction_kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  Program prog_out;
  cudaMemcpy(&prog_out, d_prog_out, sizeof(Program), cudaMemcpyDeviceToHost);

  print_program(&prog_out);

  // printf("hi\n");

  return 0;
}
