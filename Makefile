OPT=-O3
OPT_SERIAL=-O3

all: tester_parallel tester_serial

# Useful macros
#  $@: target
#  $<: first dependency
#  $^: list of all dependencies

tester_parallel: tester_parallel.o skip_parallel.o safety.o
	nvcc $(OPT) -rdc=true -arch=sm_35 -o $@ $^
tester_serial: tester_serial.o skip_serial.o
	gcc $(OPT_SERIAL) -o $@ $^

# for object files, mind the "-c"
tester_parallel.o: tester_parallel.cu skip_parallel.h Makefile
	nvcc $(OPT) -rdc=true -arch=sm_35 -c -o $@ $<
skip_parallel.o: skip_parallel.cu skip_parallel.h Makefile
	nvcc $(OPT) -rdc=true -arch=sm_35 -c -o $@ $<
safety.o: safety.cu Makefile
	nvcc $(OPT) -rdc=true -arch=sm_35 -c -o $@ $<
tester_serial.o: tester_serial.c skip_serial.h Makefile
	gcc $(OPT_SERIAL) -c -o $@ $<
skip_serial.o: skip_serial.c skip_serial.h Makefile
	gcc $(OPT_SERIAL) -c -o $@ $<

clean:
	rm -fv tester_parallel tester_serial *.o
