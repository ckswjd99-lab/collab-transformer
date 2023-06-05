#include "task.hpp"

class Worker {

private:
    TaskQueue *tq;

public:
    Worker(TaskQueue *tq);
    ~Worker();

    void run();
};