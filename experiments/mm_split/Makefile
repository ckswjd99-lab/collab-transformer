CC=gcc -g

.PHONY:

test:
	${CC} ./mm_split.c ../../libopenblas_cortexa72_st-r0.3.21.dev.a -o mm_split.out
	${CC} ./mm_nosplit.c ../../libopenblas_cortexa72_st-r0.3.21.dev.a -o mm_nosplit.out
	./mm_split.out
	./mm_nosplit.out