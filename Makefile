all: main.exe

main.exe: main.c matrix.c nn.c
	gcc -Wall -Wextra main.c matrix.c nn.c -o main.exe

clean:
	rm -rf *.o main.exe