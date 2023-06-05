CC=gcc -g

.PHONY: test

test: test_st test_mt

test_st:
	${CC} encoder.c ../../libopenblas_cortexa72_st-r0.3.21.dev.a -o encoder_st.out 

test_mt:
	${CC} encoder.c -lopenblas -o encoder_mt.out 

clean:
	rm *.out