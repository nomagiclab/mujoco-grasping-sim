#LINUX
COMMON=-O2 -I../../../include -L../../../bin -mavx -pthread -Wl,-rpath,'$$ORIGIN'
LIBS = -lmujoco210 -lGL -lm -lglew -lfreeimage ../../../bin/libglfw.so.3 `pkg-config --cflags --libs opencv`
CC = g++

ROOT = map_maker

all:
	$(CC) $(COMMON) main.cpp $(LIBS) -o ../../../bin/$(ROOT)

main.o:
	$(CC) $(COMMON) -c main.cpp

clean:
	rm *.o ../../../bin/$(ROOT)
