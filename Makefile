all: test 
test: test.c
	gcc -pthread -lm -o test test.c
clean:
	rm -f *.o test
