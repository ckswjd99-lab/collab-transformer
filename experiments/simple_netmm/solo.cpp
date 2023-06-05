#include "config.hpp"
#include "worker.hpp"


void error_handling(char *message)
{
        fputs(message, stderr);
        fputc('\n', stderr);
        exit(1);
}

int main(int argc, char **argv) {

    // init semaphores
    init_sem();

    // create workers
    pthread_t workers_thread[(WORKER_NUM)];
    int thread_id;
    void *thread_status;

    for (int i=0; i<(WORKER_NUM); i++) {
        int *worker_id = (int *)malloc(sizeof(int));
        *worker_id = i;
        if ((thread_id = pthread_create(&workers_thread[i], NULL, worker_solo, worker_id)) < 0) {
            perror("pthread_create fail!\n");
            exit(EXIT_FAILURE);
        }
    }

    // start work!

    for (int i=0; i<HEAD_NUM; i++) {
        push_workqueue(new Matrix(ENCO_QKV, i));
    }
    for (int i=0; i<SPLIT_NUM; i++) {
        push_workqueue(new Matrix(ENCO_DUMMY, i));
    }

    for (int i=0; i<(WORKER_NUM); i++) {
        push_workqueue(NULL);
    }

    for (int i=0; i<(WORKER_NUM); i++) {
        pthread_join(workers_thread[i], (void**)&thread_status);
    }
    
    return 0;
}