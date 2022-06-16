#include<catch2/catch.hpp>
#include<gvec/vec.h>

using namespace gvec;

TEST_CASE("Give_Vec_Set_Check_Get")
{
    Vec v(2); // vector of 2 elements
    v.setValue(0, 2.0f); 
    v.setValue(1, 3.0f);

    REQUIRE(v.getValue(0) == Approx(2.0f).epsilon(1e-5));
    REQUIRE(v.getValue(1) == Approx(3.0f).epsilon(1e-5));
}


TEST_CASE("Give_Vec_Init_Check_Get")
{
    Vec v = {2.0f, 3.0f};

    REQUIRE(v.getValue(0) == Approx(2.0f).epsilon(1e-5));
    REQUIRE(v.getValue(1) == Approx(3.0f).epsilon(1e-5));
}

TEST_CASE("Given_2_Vec_Check_Sum")
{
    const Vec l = {2.0f, 3.0f};
    const Vec r = {3.0f, 5.0f};

    Vec sum = l + r;

    REQUIRE(sum.size() == l.size());
    REQUIRE(sum.size() == r.size());
    REQUIRE(sum.getValue(0) == Approx(5.0f).epsilon(1e-5));
    REQUIRE(sum.getValue(1) == Approx(8.0f).epsilon(1e-5));
}
