NVCC = nvcc
override CUDAFLAGS += -std=c++17 -arch=native
# override CPPFLAGS += -std=c++20

.PHONY: all clean

all: build/hvm
clean:
	rm ./build/* -rf
	rm ./owl/*.h -f

build/hvm: build/runtime.o build/frontend.o build/hashmap.o
	$(NVCC) $(CUDAFLAGS) ./build/*.o -o ./build/hvm

build/runtime.o: src/runtime.cu ./src/common.h
	$(NVCC) $(CUDAFLAGS) -c ./src/runtime.cu -o ./build/runtime.o

build/frontend.o: src/frontend.c owl/parser.h ./src/common.h
	$(CC) $(CFLAGS) -c ./src/frontend.c -o ./build/frontend.o

build/hashmap.o: ./vendor/hashmap.c/hashmap.c
	$(CC) $(CFLAGS) -c ./vendor/hashmap.c/hashmap.c -o ./build/hashmap.o

owl/parser.h: owl/grammar.owl
	owl -c ./owl/grammar.owl -o ./owl/parser.h
