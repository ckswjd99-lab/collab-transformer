CC = g++ -g

.PHONY: server client solo

all: server client solo

server:
	$(CC) server.cpp worker.cpp ../../libopenblas_cortexa72_st-r0.3.21.dev.a -o server.out -lpthread

client:
	$(CC) client.cpp worker.cpp ../../libopenblas_cortexa72_st-r0.3.21.dev.a -o client.out -lpthread

solo:
	$(CC) solo.cpp worker.cpp ../../libopenblas_cortexa72_st-r0.3.21.dev.a -o solo.out -lpthread

copy_to_raspi_mine:
	scp ./* 192.168.1.34:~/Workspace/collab-transformer/experiments/simple_netmm/