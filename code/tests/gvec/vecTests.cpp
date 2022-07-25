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

TEST_CASE("Given_2_Vec_Check_Sub")
{
    const Vec l = {2.0f, 3.0f};
    const Vec r = {3.0f, 5.0f};

    Vec sub = l - r;

    REQUIRE(sub.size() == l.size());
    REQUIRE(sub.size() == r.size());
    REQUIRE(sub.getValue(0) == Approx(-1.0f).epsilon(1e-5));
    REQUIRE(sub.getValue(1) == Approx(-2.0f).epsilon(1e-5));
}

TEST_CASE("Given_2_Vec_Check_Mul")
{
    const Vec l = {2.0f, 3.0f};
    const Vec r = {3.0f, 5.0f};

    Vec mul = l * r;

    REQUIRE(mul.size() == l.size());
    REQUIRE(mul.size() == r.size());
    REQUIRE(mul.getValue(0) == Approx(6.0f).epsilon(1e-5));
    REQUIRE(mul.getValue(1) == Approx(15.0f).epsilon(1e-5));
}

TEST_CASE("Given_2_Vec_Check_Div")
{
    const Vec l = {2.0f, 3.0f};
    const Vec r = {3.0f, 5.0f};

    Vec div = l / r;

    REQUIRE(div.size() == l.size());
    REQUIRE(div.size() == r.size());
    REQUIRE(div.getValue(0) == Approx(2.0f/3.0f).epsilon(1e-5));
    REQUIRE(div.getValue(1) == Approx(3.0f/5.0f).epsilon(1e-5));
}

TEST_CASE("Given_2_Vec_Dot")
{
    const Vec l = {2.0f, 3.0f};
    const Vec r = {3.0f, 5.0f};

    float dot_product = l.dot(r);

    REQUIRE(dot_product == Approx(2.0f*3.0f + 3.0f*5.0f).epsilon(1e-5));
}
