#include "worker.hpp"
#include <pthread.h>

void *work(void *arg) {
    TaskQueue *tq = (TaskQueue *)arg;
    Worker *worker = new Worker(tq);
    worker->run();
    delete worker;

    return NULL;
}

int main() {

    TaskQueue *tq = new TaskQueue(1);

    pthread_t threads[3];

    int thread_id, status;

    if ((thread_id = pthread_create(&threads[0], NULL, work, tq)) < 0) {
        perror("thread error!\n");
    }
    if ((thread_id = pthread_create(&threads[1], NULL, work, tq)) < 0) {
        perror("thread error!\n");
    }
    if ((thread_id = pthread_create(&threads[2], NULL, work, tq)) < 0) {
        perror("thread error!\n");
    }

    pthread_join(threads[0], (void **)&status);
    pthread_join(threads[1], (void **)&status);
    pthread_join(threads[2], (void **)&status);

    delete tq;

    return 0;
}