#include "task.hpp"

int main () {

    TaskQueue *tq = new TaskQueue(1);
    tq->print();

    tq->pop_front();
    tq->print();

    tq->push_back(new Task(1, 0));
    tq->print();

    tq->push_front(new Task(1, 1));
    tq->print();

    return 0;
}