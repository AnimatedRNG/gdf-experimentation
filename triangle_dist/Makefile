default: triangle_distance triangle_distance.so

triangle_distance: triangle_distance.c
	gcc -g -Wall -Wextra -pedantic -Wfatal-errors -msse4.2 -O2 -lm -lgcc triangle_distance.c -o triangle_distance

triangle_distance.so: triangle_distance.c
	gcc -g -Wall -Wextra -pedantic -Wfatal-errors -msse4.2 -O2 -lm -lgcc triangle_distance.c -fPIC -rdynamic -shared -Lstatic -o triangle_distance.so

clean:
	-rm -f triangle_distance triangle_distance.so
