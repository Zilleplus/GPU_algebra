namespace gvec{
    class Vec{
        float* d_;
        int size_;

        public:
            Vec(int size);
            Vec(const std::initializer_list<float>& elements);
            Vec(Vec&& other);

            ~Vec();

            int size() const;

            float getValue(int i) const;

            void setValue(int index, float value);

            Vec operator+(const Vec& other) const;
            Vec operator-(const Vec& other) const;
            Vec operator*(const Vec& other) const;
            Vec operator/(const Vec& other) const;
            float dot(const Vec& other) const ;

            static constexpr int threadsPerBlock = 16;

            int numBlocks() const {
                const int numBlocks = 
                    std::min(32, (size_ + threadsPerBlock-1)/threadsPerBlock);
                return numBlocks;
            }
    };
}
