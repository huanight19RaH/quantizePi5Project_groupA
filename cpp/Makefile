CXX ?= g++
CXXFLAGS := -O3 -std=c++17 -mcpu=cortex-a76+fp16+dotprod -DNDEBUG -Wall -Wextra -Wno-missing-field-initializers
LDFLAGS  := -lrknnrt -lrga -lpthread

all: libyolo26_rk3588.so yolo26_edge_cli

libyolo26_rk3588.so: yolo26_rk3588.cpp yolo26_rk3588.h
	$(CXX) $(CXXFLAGS) -fPIC -shared -DBUILD_SHARED_LIB yolo26_rk3588.cpp $(LDFLAGS) -o $@

yolo26_edge_cli: yolo26_rk3588.cpp yolo26_rk3588.h yolo26_edge_cli.cpp
	$(CXX) $(CXXFLAGS) yolo26_rk3588.cpp yolo26_edge_cli.cpp $(LDFLAGS) -o $@

clean:
	rm -f libyolo26_rk3588.so yolo26_edge_cli
