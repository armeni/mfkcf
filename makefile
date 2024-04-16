.DEFAULT_GOAL := all

USE_VOT_TRAX=0

CC=gcc
CXX=g++

# Use pkg-config to get OpenCV flags
LDFLAGS= $(shell pkg-config --libs opencv4) -lstdc++ -lm
CXXFLAGS= -Wall $(shell pkg-config --cflags opencv4) -std=c++17 -lstdc++ -O3 -fPIC
HEADERS = $(wildcard *.h) $(wildcard *.hpp)
TARGET_LIB = libkcftracker.so
OBJS = fhog.o kcftracker.o mixformer_trt.o
LDFLAGS += -L/usr/local/cuda-12.0 -lcudart -lcuda -lnvinfer

ALL = track.bin $(TARGET_LIB)

all: $(ALL)
	@$(MAKE) clean

track.bin: $(OBJS) runkcftracker.o
	$(CC) -o $@ $^ $(LDFLAGS)

$(TARGET_LIB): $(OBJS)
	$(CC) ${LDFLAGS} -shared -o $@ $^

%.o: %.c $(HEADERS)
	$(CC) -c -o $@ $< $(CFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

.PHONY: clean

clean:
	rm -rf *.o *.so

.PHONY: install
install: $(TARGET_LIB)
	mkdir -p /usr/local/include/mfkcf
	mkdir -p /usr/local/include/mfkcf
	cp $(TARGET_LIB) /usr/local/lib
	mkdir -p /usr/local/include/mfkcf
	cp *.hpp /usr/local/include/mfkcf

.PHONY: uninstall
uninstall: $(TARGET_LIB)
	rm -f -r /usr/local/include/mfkcf
	rm -f /usr/local/lib/$(TARGET_LIB)

