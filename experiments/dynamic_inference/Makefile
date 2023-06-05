CC=g++ -g

.PHONY: test_task test_worker

test_task:
	${CC} task.cpp test_task.cpp ../../libopenblas_cortexa72_st-r0.3.21.dev.a -lpthread -o test_task.out

test_worker:
	${CC} task.cpp worker.cpp test_worker.cpp ../../libopenblas_cortexa72_st-r0.3.21.dev.a -lpthread -o test_worker.out