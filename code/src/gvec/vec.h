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

            Vec operator+(const Vec& other) const &;
    };
}
