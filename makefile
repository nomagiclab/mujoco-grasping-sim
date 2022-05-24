
#MAC
#COMMON=-O2 -I../../include -L../../bin -mavx -pthread
#LIBS = -w -lmujoco200 -lglfw.3
#CC = gcc

#LINUX
COMMON=-O2 -I../../include -L../../bin -mavx -pthread -Wl,-rpath,'$$ORIGIN' -g
LIBS = -lmujoco200 -lGL -lm -lglew -lfreeimage ../../bin/libglfw.so.3 `pkg-config --cflags --libs opencv`
CC = g++

#WINDOWS
#COMMON=/O2 /MT /EHsc /arch:AVX /I../../include /Fe../../bin/
#LIBS = ../../bin/glfw3.lib  ../../bin/mujoco200.lib
#CC = cl

ROOT = mujoco-grasping-sim

all:
	$(CC) $(COMMON) main.cpp $(LIBS) -o ../../bin/$(ROOT)

main.o:
	$(CC) $(COMMON) -c main.cpp

clean:
	rm *.o ../../bin/$(ROOT)
