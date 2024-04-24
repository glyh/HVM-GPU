#define OWL_PARSER_IMPLEMENTATION
#include "../owl/parser.h"
#include "../vendor/hashmap.c/hashmap.h"
#include "common.h"

#define SEED0 12991981
#define SEED1 82381933

// NOTE: Nvidia, fuck you for forcing me to write non-ideal code
// Port: Constructor and Getters
// -----------------------------
static inline Port new_port(Tag tag, Val val) {
  return (val << TAG_BITS) | tag;
}
static inline Port new_var(Val val) { return new_port(VAR, val); }
static inline Port new_ref(Val val) { return new_port(REF, val); }
static inline Port new_num(Val val) { return new_port(NUM, val); }
static inline Port new_era() { return new_port(ERA, 0); }
static inline Port new_con(Val val) { return new_port(CON, val); }
static inline Port new_dup(Val val) { return new_port(DUP, val); }
static inline Port new_opr(Val val) { return new_port(OPR, val); }
static inline Port new_swi(Val val) { return new_port(SWI, val); }
static inline Port new_nil() { return 0; }
static inline Tag get_tag(Port port) { return port & TAG_MASK; }
static inline Val get_val(Port port) { return port >> TAG_BITS; }
// Pair: Constructor and Getters
// -----------------------------
static inline Pair new_pair(Port fst, Port snd) {
  return ((u32)fst << 16) | snd;
}

static inline void panic(const char *format, ...) {
  va_list argptr;
  va_start(argptr, format);
  vfprintf(stderr, format, argptr);
  va_end(argptr);
  exit(1);
}

struct ref {
  struct parsed_identifier id;
  int uid;
};

uint64_t ref_hash(const void *_item, uint64_t seed0, uint64_t seed1) {
  const struct ref *item = _item;
  return hashmap_xxhash3(item->id.identifier, item->id.length, seed0, seed1);
}

int ref_compare(const void *_a, const void *_b, void *udata) {
  const struct ref *a = _a;
  const struct ref *b = _b;
  const char *id_a = a->id.identifier;
  const char *id_b = b->id.identifier;
  size_t offset = 0;
  while (*id_a == *id_b && offset < a->id.length && offset < b->id.length) {
    ++id_a;
    ++id_b;
  }
  if (*id_a != *id_b) {
    return *id_a - *id_b;
  } else {
    return a->id.length - b->id.length;
  }
}

bool hashmap_ensure(struct hashmap *map, const void *item) {
  const void *got = hashmap_get(map, item);
  if (got == NULL) {
    hashmap_set(map, item);
    return true;
  }
  return false;
}

struct hashmap *ref_map, *current_var_map;
void ensure_ref(struct parsed_identifier id, int *uid_counter) {
  struct ref *item = malloc(sizeof(struct ref));
  item->id = id;
  item->uid = *uid_counter;
  if (hashmap_ensure(ref_map, item)) {
    ++*uid_counter;
  }
}

int ref_counter = 0;
struct Def *current_def = NULL;
Port register_term(struct parsed_term term) {
  Port inners[2];
  Val port_id;
  unsigned term_id;
  switch (term.type) {
  case PARSED_ERA:
    return new_nil();
  case PARSED_CON:
  case PARSED_DUP:
  case PARSED_SWI: {
    term_id = 0;
    for (; !term.term.empty; term.term = owl_next(term.term)) {
      struct parsed_term term_inner = parsed_term_get(term.term);
      printf("%lu %lu\n", term_inner.range.start, term_inner.range.end);
      inners[term_id++] = register_term(term_inner);
    }
    assert(term_id == 2);
    port_id = current_def->_NODE_LEN++;
    current_def->NODE_BUF[port_id] = new_pair(inners[0], inners[1]);
    Tag tag;
    switch (term.type) {
    case PARSED_CON:
      tag = CON;
      break;
    case PARSED_DUP:
      tag = DUP;
      break;
    case PARSED_SWI:
      tag = SWI;
      break;
    default:
      panic("Unreachable");
    }
    return new_port(tag, port_id);
  }
  // TODO: maybe smaller nums can be optimized with one less re-link
  // but for u60 definitely a relink is needed
  // for now we only takes care of u(16-3) = u13
  case PARSED_NUM: {
    double num = parsed_number_get(term.number).number;
    if (num < 0 || num >= (1 << VAL_BITS)) {
      panic("%lu has too many bits to be compiled as a NUM\n", num);
    }
    Val num_ranged = num;
    return new_num(num_ranged);
  }
  case PARSED_OP2: {
    panic("Unimplemented OP2");
  }
  case PARSED_VAR: {
    struct parsed_identifier var_id = parsed_identifier_get(term.identifier);
  }
  default:
    return 0;
  }
}
struct Def read_definition(struct parsed_def def) {
  struct parsed_identifier id =
      parsed_identifier_get(parsed_ref_get(def.ref).identifier);
  ensure_ref(id, &ref_counter);
  struct parsed_term primary_term = parsed_term_get(def.term);
  struct Def ret = {};
  ret._NODE_LEN = 1; // reserve space for root node
  current_def = &ret;
  ret.NODE_BUF[0] = register_term(primary_term);
  return ret;
}

struct Book parse() {
  struct Book ret;

  struct owl_tree *tree;
  ref_map = hashmap_new(sizeof(struct ref), 100, SEED0, SEED1, ref_hash,
                        ref_compare, NULL, NULL);
  tree = owl_tree_create_from_file(stdin);
  // owl_tree_print(tree);
  struct parsed_book book = owl_tree_get_parsed_book(tree);
  u32 uid = 0;
  for (; !book.def.empty; book.def = owl_next(book.def)) {
    struct parsed_def def = parsed_def_get(book.def);
    ret.DEFS_BUF[ret.DEFS_LEN++] = read_definition(def);
  }
  owl_tree_destroy(tree);
  hashmap_free(ref_map);
  return ret;
}

int main() {
  parse();
  return 0;
}
