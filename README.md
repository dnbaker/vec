# vec
Type-generic SIMD library for optimized generic code generation

## Usage

For simple operations, one can simple select `block_apply(begin, end, functor)`, where functor is selected from `SIMDTypes<float_type>::apply_##op##_##prec`, where op is the operation and prec is the precision. Not all operations support all of u05, u10, and u35, so you can either examine the generated header file from SLEEF or work by trial and error.

For more complex operations, one can access the appropriate assembly instructions and work using the types and instructions contained in `SIMDTypes<float_type>`. For an example of this kind of use, see [this Gaussian finalizer from a Fast Random Fourier Features implementation.](https://github.com/dnbaker/frp/blob/master/include/frp/kernel.h#L10-L72)

## TODO:

Port the remainder of sleef.
This includes:
tan
a{sin,tan,cos}
{sin,tan,cos}h
a{sin,tan,cos}h
exp{,2,10,m1}
log{2,10,1p)
{ld,fr}exp
fma
sqrt
hypot
fabs
copysign
fm{ax,in}
fdimc
ceil
round
floor
{t,l}gamma
erf{,c}
log
cbrt
pow
