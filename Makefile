all: test 
test: test.c
	gcc -Wall -pthread -lrt -lm -o test test.c
clean:
	rm -f *.o test
