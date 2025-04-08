#pragma once
#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <atomic>

class ThreadPool
{
public:
      ThreadPool() {  }
//    ThreadPool() {  }
     ~ThreadPool() {  }
//   ~ThreadPool() {  }

    void WarmUp()
//  void WarmUp()
    {
        size_t numberOfThreadsAvailable = std::max<size_t>(1, std::thread::hardware_concurrency());
//      size_t numberOfThreadsAvailable = std::max<size_t>(1, std::thread::hardware_concurrency());
        for (size_t i = 0; i < numberOfThreadsAvailable; ++i)
        {
            threads.emplace_back([this]
            {
                while (true)
                {
                    std::function<void()> task;
//                  std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(tasksMutex);
//                      std::unique_lock<std::mutex> lock(tasksMutex);
                        conditionVariableForNewTasks.wait(lock, [this] { return !tasks.empty() || stop; });
//                      conditionVariableForNewTasks.wait(lock, [this] { return !tasks.empty() || stop; });
                        if (stop && tasks.empty()) { return; }
//                      if (stop && tasks.empty()) { return; }
                        task = std::move(tasks.front());
//                      task = std::move(tasks.front());
                        tasks.pop();
//                      tasks.pop();
                        ++currentlyExecutingTasksCount;
//                      ++currentlyExecutingTasksCount;
                    }
                    task();
//                  task();
                    {
                        std::unique_lock<std::mutex> lock(tasksMutex);
//                      std::unique_lock<std::mutex> lock(tasksMutex);
                        --currentlyExecutingTasksCount;
//                      --currentlyExecutingTasksCount;
                        if (currentlyExecutingTasksCount == 0 && tasks.empty())
//                      if (currentlyExecutingTasksCount == 0 && tasks.empty())
                        {
                            conditionVariableForAllTasks.notify_all();
//                          conditionVariableForAllTasks.notify_all();
                        }
                    }
                }
            });
        }
    }

    void WarmUp(size_t numberOfThreads)
//  void WarmUp(size_t numberOfThreads)
    {
        size_t numberOfThreadsAvailable = std::min<size_t>(numberOfThreads, std::thread::hardware_concurrency());
//      size_t numberOfThreadsAvailable = std::min<size_t>(numberOfThreads, std::thread::hardware_concurrency());
        for (size_t i = 0; i < numberOfThreadsAvailable; ++i)
        {
            threads.emplace_back([this]
            {
                while (true)
                {
                    std::function<void()> task;
//                  std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(tasksMutex);
//                      std::unique_lock<std::mutex> lock(tasksMutex);
                        conditionVariableForNewTasks.wait(lock, [this] { return !tasks.empty() || stop; });
//                      conditionVariableForNewTasks.wait(lock, [this] { return !tasks.empty() || stop; });
                        if (stop && tasks.empty()) { return; }
//                      if (stop && tasks.empty()) { return; }
                        task = std::move(tasks.front());
//                      task = std::move(tasks.front());
                        tasks.pop();
//                      tasks.pop();
                        ++currentlyExecutingTasksCount;
//                      ++currentlyExecutingTasksCount;
                    }
                    task();
//                  task();
                    {
                        std::unique_lock<std::mutex> lock(tasksMutex);
//                      std::unique_lock<std::mutex> lock(tasksMutex);
                        --currentlyExecutingTasksCount;
//                      --currentlyExecutingTasksCount;
                        if (currentlyExecutingTasksCount == 0 && tasks.empty())
//                      if (currentlyExecutingTasksCount == 0 && tasks.empty())
                        {
                            conditionVariableForAllTasks.notify_all();
//                          conditionVariableForAllTasks.notify_all();
                        }
                    }
                }
            });
        }
    }

    void WaitForAllTasksToBeDone()
//  void WaitForAllTasksToBeDone()
    {
        std::unique_lock<std::mutex> lock(tasksMutex);
//      std::unique_lock<std::mutex> lock(tasksMutex);
        conditionVariableForAllTasks.wait(lock, [this] { return tasks.empty() && currentlyExecutingTasksCount == 0; });
//      conditionVariableForAllTasks.wait(lock, [this] { return tasks.empty() && currentlyExecutingTasksCount == 0; });
    }

    void Stop()
//  void Stop()
    {
        {
            std::unique_lock<std::mutex> lock(tasksMutex);
//          std::unique_lock<std::mutex> lock(tasksMutex);
            stop = true;
//          stop = true;
        }
        conditionVariableForNewTasks.notify_all();
//      conditionVariableForNewTasks.notify_all();

        for (std::thread& thread : threads)
        {
            if (thread.joinable())
//          if (thread.joinable())
            {
                thread.join();
//              thread.join();
            }
        }
    }

    void Enqueue(std::function<void()>&& task)
//  void Enqueue(std::function<void()>&& task)
    {
        {
            std::unique_lock<std::mutex> lock(tasksMutex);
//          std::unique_lock<std::mutex> lock(tasksMutex);
            tasks.emplace(std::move(task));
//          tasks.emplace(std::move(task));
        }
        conditionVariableForNewTasks.notify_one();
//      conditionVariableForNewTasks.notify_one();
    }

    ThreadPool           (const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ThreadPool           (ThreadPool&&) = default;
    ThreadPool& operator=(ThreadPool&&) = default;

private:
    std::vector<std::thread> threads;
//  std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
//  std::queue<std::function<void()>> tasks;
    std::mutex tasksMutex;
//  std::mutex tasksMutex;
    std::condition_variable conditionVariableForNewTasks;
//  std::condition_variable conditionVariableForNewTasks;
    std::condition_variable conditionVariableForAllTasks;
//  std::condition_variable conditionVariableForAllTasks;
    std::atomic<std::size_t> currentlyExecutingTasksCount = 0;
//  std::atomic<std::size_t> currentlyExecutingTasksCount = 0;
    std::atomic<bool> stop = false;
//  std::atomic<bool> stop = false;
};

#endif
