#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#pragma once

// Integers
// --------

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef unsigned long long int u64;

// Configuration
// -------------

const u32 SMS = 1; // Processor Count
const u32 TPB = 1; // Threads Per Processor

// Types
// -----

typedef u8 Tag;   // tag: 3-bit (rounded up to u8)
typedef u16 Val;  // val: 13-bit (rounded up to u16)
typedef u16 Port; // port: tag + val (fits an u16 exactly)
typedef u32 Pair; // pair: port + port
#define TAG_BITS 3
#define TAG_MASK ((1 << TAG_BITS) - 1)
#define VAL_BITS (sizeof(Val) * (CHAR_BIT))

// Tags
// NUM takes 0x0 so it's easy to do numeric operation on U60s
const Tag NUM = 0x0; // number
const Tag VAR = 0x1; // variable
const Tag REF = 0x2; // reference
const Tag ERA = 0x3; // eraser
const Tag CON = 0x4; // constructor
const Tag DUP = 0x5; // duplicator
const Tag OPR = 0x6; // operation (binary)
const Tag SWI = 0x7; // switch

// Port Number
const Tag P1 = 0; // PORT-1
const Tag P2 = 1; // PORT-2

// Interaction Rule Type
typedef u8 Rule;

// Interaction Rule Values
const Rule LINK = 0x0;
const Rule VOID = 0x1;
const Rule ERAS = 0x2;
const Rule ANNI = 0x3;
const Rule COMM = 0x4;
const Rule OPER = 0x5;
const Rule SWIT = 0x6;
const Rule CALL = 0x7;
const Rule SWAP = 0x8;
const Rule NOOP = 0x9;

// Program
#define RBAG_LEN 0x1000 // max 4096 redexes
#define NODE_LEN 0x2000 // max 8196 nodes
#define VARS_LEN 0x2000 // max 8196 vars
struct Program {
  u64 rewrites;
  u32 RBAG_BUF[RBAG_LEN];
  u32 NODE_BUF[NODE_LEN];
  u32 VARS_BUF[VARS_LEN]; // NOTE: can't use u16 due to no atomicExch
};

// Definition
struct Def {
  u32 MASK; // Nth bit set when node N is numeric
  u32 _RBAG_LEN;
  Pair RBAG_BUF[32];
  u32 _NODE_LEN;
  Pair NODE_BUF[32];
  u32 _VARS_LEN;
};

// Book
struct Book {
  u32 DEFS_LEN;
  struct Def DEFS_BUF[128];
};

// Local Redex Bag
// It uses the same space to store two stacks:
// - HI: a high-priotity stack, for shrinking reductions
// - LO: a low-priority stack, for growing reductions
struct RBag {
  u32 hi_idx;  // high-priority stack push-index
  u32 lo_idx;  // low-priority stack push-index
  u32 buf[32]; // a shared buffer for both stacks
};

// None
const u32 NONE = 0xFFFFFFFF;
