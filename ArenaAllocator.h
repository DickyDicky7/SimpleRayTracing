#pragma once
#ifndef ARENA_ALLOCATOR_H
#define ARENA_ALLOCATOR_H



  #include <cstddef>
//#include <cstddef>
  #include <stdexcept>
//#include <stdexcept>
  #include <new> // For placement new
//#include <new> // For placement new
  #include <utility> // For std::forward
//#include <utility> // For std::forward



class ArenaAllocator
{
public:
    ArenaAllocator(size_t size) : size_(size), offset_(0)
//  ArenaAllocator(size_t size) : size_(size), offset_(0)
    {
        buffer_ = new char[size];
//      buffer_ = new char[size];
    }

    // Delete copy constructor and assignment to prevent shallow copies
    // Delete copy constructor and assignment to prevent shallow copies
    ArenaAllocator           (const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    // Optional: Move constructor could be implemented if needed
    // Optional: Move constructor could be implemented if needed
    ArenaAllocator           (ArenaAllocator&&) = default;
    ArenaAllocator& operator=(ArenaAllocator&&) = default;

    ~ArenaAllocator()
//  ~ArenaAllocator()
    {
        delete[] buffer_;
//      delete[] buffer_;
    }

    void* Allocate(size_t bytes)
//  void* Allocate(size_t bytes)
    {
        // Align the offset
        // Align the offset
        size_t alignment = alignof(std::max_align_t);
//      size_t alignment = alignof(std::max_align_t);
        size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);
//      size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);

        if (aligned_offset + bytes > size_)
//      if (aligned_offset + bytes > size_)
        {
            throw std::bad_alloc();
//          throw std::bad_alloc();
        }

        void* ptr = buffer_ + aligned_offset;
//      void* ptr = buffer_ + aligned_offset;
        offset_ = aligned_offset + bytes;
//      offset_ = aligned_offset + bytes;
        return ptr;
//      return ptr;
    }

    void* Allocate(size_t bytes, size_t alignment)
//  void* Allocate(size_t bytes, size_t alignment)
    {
        size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);
//      size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);
        if (aligned_offset + bytes > size_) throw std::bad_alloc();
//      if (aligned_offset + bytes > size_) throw std::bad_alloc();
        void* ptr = buffer_ + aligned_offset;
//      void* ptr = buffer_ + aligned_offset;
        offset_ = aligned_offset + bytes;
//      offset_ = aligned_offset + bytes;
        return ptr;
//      return ptr;
    }

    void Deallocate(void* /*ptr*/)
//  void Deallocate(void* /*ptr*/)
    {
        // No-op; arena allocator doesn't support individual deallocation
//      // No-op; arena allocator doesn't support individual deallocation
    }

    void Reset()
//  void Reset()
    {
        offset_ = 0;
//      offset_ = 0;
    }

    template<typename T, typename... Args>
//  template<typename T, typename... Args>
    T* ConstructGenerals(Args&&... args)
//  T* ConstructGenerals(Args&&... args)
    {
        void* ptr = Allocate(sizeof(T));
//      void* ptr = Allocate(sizeof(T));
        return new(ptr) T(std::forward<Args>(args)...); // Placement new to construct the object
//      return new(ptr) T(std::forward<Args>(args)...); // Placement new to construct the object
    }

    template<typename T, typename... Args>
//  template<typename T, typename... Args>
    T* ConstructSpecific(Args&&... args)
//  T* ConstructSpecific(Args&&... args)
    {
        void* ptr = Allocate(sizeof(T), alignof(T));
//      void* ptr = Allocate(sizeof(T), alignof(T));
        return new(ptr) T(std::forward<Args>(args)...); // Placement new to construct the object
//      return new(ptr) T(std::forward<Args>(args)...); // Placement new to construct the object
    }

private:
    char* buffer_;
//  char* buffer_;
    size_t size_;
//  size_t size_;
    size_t offset_;
//  size_t offset_;
};



#endif


















