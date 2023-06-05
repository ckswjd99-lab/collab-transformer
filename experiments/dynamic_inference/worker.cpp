#include "worker.hpp"

Worker::Worker(TaskQueue *tq) {
    this->tq = tq;
}

Worker::~Worker() { }

void Worker::run() {
    Task *task;
    
    while(task = this->tq->pop_front()) {
        printf("task (%d, %d)\n", task->get_layer(), task->get_op_idx());
        task->compute();
        if (task->get_layer() == 0) {
            this->tq->push_front(new Task(1, task->get_op_idx()));
        }
        delete task;
    }
}