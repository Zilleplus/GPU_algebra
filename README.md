# GPU_algebra: Simple Vector algebra library in C++ using CUDA

## Goal
- Learn how cmake works with cuda.
- Learn basic cuda concepts.
- Learn how to debug cuda code.

## General setup
- simple cmake/conan build. (build on ubuntu 20.10 with cuda compiler 11.6-r11.6, ran on a nvidea 3060)
- 1 vector class with a header without any cuda based includes.
- cach2 tests.

## API
- Vec.h
```
    class Vec{
        float* d_;
        int size_;

        public:
            Vec(int size);
            Vec(const std::initializer_list<float>& elements);
            Vec(Vec&& other);

            ~Vec();

            int size() const;

            float getValue(int index) const;

            void setValue(int index, float value);

            Vec operator+(const Vec& other) const;
            Vec operator-(const Vec& other) const;
            Vec operator*(const Vec& other) const;
            Vec operator/(const Vec& other) const;
            float dot(const Vec& other) const ;

            static constexpr int threadsPerBlock = 16;
            int numBlocks() const;
    };

```
### Operations:
- Vec(int size): create vector of size "size".
- Vec(const std::initialize_list<float>&): create vector with elements from inialize_list.
- Vec(Vec&&): move constructor.
- getValue(int index): get one value from the gpu memory.
- setValue(int index, float value): set one value on the gpu memory.
- The operators +-*/ are all element wise operations.
- dot: dotproduct of 2 vectors (reduce operation).