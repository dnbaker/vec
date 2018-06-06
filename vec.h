#ifndef _VEC_H__
#define _VEC_H__
#define NOSVML
#ifndef NO_SLEEF
#  include "sleef/include/sleefdft.h"
#  include "sleef.h"
#endif // #ifndef NO_SLEEF
#include "x86intrin.h"
#include <cmath>
#include <iterator>
#include <type_traits>
#include <cstdint>
#include <array>
#ifndef NO_BLAZE
#include "blaze/Math.h"
#endif

#ifndef IS_BLAZE
#  define IS_BLAZE(x) (::blaze::IsVector<x>::value || ::blaze::IsMatrix<x>::value)
#endif
#ifndef IS_CONTIGUOUS_BLAZE
#  define IS_CONTIGUOUS_BLAZE(x) (bool(::blaze::TransposeFlag<x>::value))
#endif
#ifndef IS_COMPRESSED_BLAZE
#  define IS_COMPRESSED_BLAZE(x) (::blaze::IsSparseVector<x>::value || ::blaze::IsSparseMatrix<x>::value)
#endif
#ifndef IS_CONTIGUOUS_UNCOMPRESSED_BLAZE
#  define IS_CONTIGUOUS_UNCOMPRESSED_BLAZE(x) (IS_BLAZE(x) && !IS_COMPRESSED_BLAZE(x) && IS_CONTIGUOUS_BLAZE(x))
#endif
#ifndef HAS_AVX_512
#  define HAS_AVX_512 (_FEATURE_AVX512F || _FEATURE_AVX512ER || _FEATURE_AVX512PF || _FEATURE_AVX512CD || __AVX512BW__ || __AVX512CD__ || __AVX512F__)
#endif

namespace vec {

using std::uint64_t;
#ifndef NO_SLEEF
namespace scalar {
    using namespace std;
    Sleef_double2 sincos(double x) {
        return Sleef_double2{sin(x), cos(x)};
    }
    Sleef_float2 sincos(float x) {
        return Sleef_float2{sin(x), cos(x)};
    }
    template<typename T> auto sqrt_u35(T val) {return sqrt(val);}
    template<typename T> auto sqrt_u05(T val) {return sqrt(val);}
}
#endif // #ifndef NO_SLEEF

template<typename ValueType>
struct SIMDTypes;

#define OP(op, suf, sz) _mm##sz##_##op##_##suf
#define decop(op, suf, sz) static constexpr decltype(&OP(op, suf, sz)) op = &OP(op, suf, sz);

/* Use or separately because it's a keyword.*/

#define declare_all(suf, sz) \
   decop(loadu, suf, sz) \
   decop(storeu, suf, sz) \
   decop(load, suf, sz) \
   decop(store, suf, sz) \
   static constexpr decltype(&_mm##sz##_cmp_##suf##_mask) cmp_mask_fn = &_mm##sz##_cmp_##suf##_mask; \
   static constexpr decltype(&OP(or, suf, sz)) or_fn = &OP(or, suf, sz);\
   static constexpr decltype(&OP(and, suf, sz)) and_fn = &OP(and, suf, sz);\
   decop(add, suf, sz) \
   decop(sub, suf, sz) \
   decop(mul, suf, sz) \
   decop(set1, suf, sz) \
   /*decop(setr, suf, sz) */\
   decop(set, suf, sz) \
   decop(mask_and, suf, sz) \
   decop(maskz_and, suf, sz) \
   decop(maskz_andnot, suf, sz) \
   decop(mask_andnot, suf, sz) \
   decop(andnot, suf, sz) \
   /*decop(blendv, suf, sz) */

#define declare_int_ls(suf, sz) \
    decop(loadu, si##sz, sz) \
    decop(load, si##sz, sz) \
    decop(storeu, si##sz, sz) \
    decop(store, si##sz, sz)

#define declare_int_ls128(suf, sz) \
    decop(loadu, si128, sz) \
    decop(load, si128, sz) \
    decop(storeu, si128, sz) \
    decop(store, si128, sz)

#define declare_int_epi64(sz) \
    decop(slli, epi64, sz) \
    decop(srli, epi64, sz) \
    decop(add, epi64, sz) \
    decop(sub, epi64, sz) \
    decop(mullo, epi64, sz) \
    static constexpr decltype(&OP(xor, si##sz, sz)) xor_fn = &OP(xor, si##sz, sz);\
    static constexpr decltype(&OP(or, si##sz, sz))  or_fn = &OP(or, si##sz, sz);\
    static constexpr decltype(&OP(and, si##sz, sz)) and_fn = &OP(and, si##sz, sz);\
    decop(set1, epi64x, sz)

#define declare_int_epi64_512(sz) \
    decop(slli, epi64, sz) \
    decop(srli, epi64, sz) \
    decop(add, epi64, sz) \
    decop(sub, epi64, sz) \
    decop(mullo, epi64, sz) \
    static constexpr decltype(&OP(xor, si##sz, sz)) xor_fn = &OP(xor, si##sz, sz);\
    static constexpr decltype(&OP(or, si##sz, sz))  or_fn = &OP(or, si##sz, sz);\
    static constexpr decltype(&OP(and, si##sz, sz)) and_fn = &OP(and, si##sz, sz);\
    decop(set1, epi64, sz)

#define declare_int_epi64_128(sz) \
    decop(slli, epi64, sz) \
    decop(srli, epi64, sz) \
    decop(add, epi64, sz) \
    decop(sub, epi64, sz) \
    decop(mullo, epi64, sz) \
    static constexpr decltype(&OP(xor, si128, sz)) xor_fn = &OP(xor, si128, sz);\
    static constexpr decltype(&OP(and, si128, sz)) and_fn = &OP(and, si128, sz);\
    static constexpr decltype(&OP(or, si128, sz))  or_fn = &OP(or, si128, sz);\
    decop(set1, epi64x, sz)

#define declare_all_int(suf, sz) \
    declare_int_ls(suf, sz) \
    declare_int_epi64(sz)

#define declare_all_int128(suf, sz) \
    declare_int_ls128(suf, sz) \
    declare_int_epi64_128(sz)

#define declare_all_int512(suf, sz) \
    declare_int_ls(suf, sz) \
    declare_int_epi64_512(sz)


#ifndef NO_SLEEF

#define SLEEF_OP(op, suf, prec, set) Sleef_##op##suf##_##prec##set
#define dec_sleefop_prec(op, suf, prec, instructset) \
    static constexpr decltype(&SLEEF_OP(op, suf, prec, instructset)) op##_##prec = \
    &SLEEF_OP(op, suf, prec, instructset); \
    struct apply_##op##_##prec {\
        template<typename... T>\
        auto operator()(T &&...args) const {return op##_##prec(std::forward<T...>(args)...);} \
        template<typename OT>\
        OT scalar(OT val) const {return scalar::op(val);} \
    };


#define dec_all_precs(op, suf, instructset) \
    dec_sleefop_prec(op, suf, u35, instructset) \
    dec_sleefop_prec(op, suf, u10, instructset)

#define dec_all_precs_u05(op, suf, instructset) \
    dec_sleefop_prec(op, suf, u35, instructset) \
    dec_sleefop_prec(op, suf, u05, instructset)


#define dec_double_sz(type) using TypeDouble = Sleef_##type##_2;


#define dec_all_trig(suf, set) \
   dec_all_precs(sin, suf, set) \
   dec_all_precs(cos, suf, set) \
   dec_all_precs(asin, suf, set) \
   dec_all_precs(acos, suf, set) \
   dec_all_precs(atan, suf, set) \
   dec_all_precs(atan2, suf, set) \
   dec_all_precs(cbrt, suf, set) \
   dec_all_precs(sincos, suf, set) \
   dec_sleefop_prec(log, suf, u10, set) \
   dec_sleefop_prec(log1p, suf, u10, set) \
   dec_sleefop_prec(expm1, suf, u10, set) \
   dec_sleefop_prec(exp, suf, u10, set) \
   dec_sleefop_prec(exp2, suf, u10, set) \
   /*dec_sleefop_prec(exp10, suf, u10, set) */ \
   dec_sleefop_prec(lgamma, suf, u10, set) \
   dec_sleefop_prec(tgamma, suf, u10, set) \
   dec_sleefop_prec(sinh, suf, u10, set) \
   dec_sleefop_prec(cosh, suf, u10, set) \
   dec_sleefop_prec(asinh, suf, u10, set) \
   dec_sleefop_prec(acosh, suf, u10, set) \
   dec_sleefop_prec(tanh, suf, u10, set) \
   dec_sleefop_prec(atanh, suf, u10, set) \
   dec_all_precs_u05(sqrt, suf, set)

#endif // #ifndef NO_SLEEF
    
template<typename SType>
union UType {
    using ValueType = typename SType::ValueType;
    using Type      = typename SType::Type;
    static constexpr size_t COUNT = SType::COUNT;
    std::array<ValueType, COUNT> arr_;
    Type                        simd_;
    UType(Type val): simd_(val) {}
    UType(ValueType val): simd_(SType::set1(val)) {}
    UType() {}
    UType &operator=(Type val) {
        simd_ = val;
        return *this;
    }
    UType &operator=(ValueType val) {
        simd_ = SType::set1(val);
        return *this;
    }
    UType &operator+=(Type val) {
        simd_ += val;
        return *this;
    }
    UType &operator+=(ValueType val) {
        simd_ += SType::set1(val);
        return *this;
    }
    UType &operator-=(Type val) {
        simd_ -= val;
        return *this;
    }
    UType &operator-=(ValueType val) {
        simd_ -= SType::set1(val);
        return *this;
    }
    template<size_t nleft, size_t done>
    struct unroller {
        UType &ref_;
        template<typename Functor>
        void for_each(const Functor &func) {
            func(ref_.arr_[COUNT - nleft]);
            unroller<nleft - 1, done + 1> ur(ref_);
            ur.for_each(func);
        }
        unroller(UType &ref): ref_(ref) {}
    };
    template<size_t done>
    struct unroller<0, done> {
        template<typename Functor> void for_each(const Functor &func) {}
        unroller(UType &ref) {}
    };
    template<size_t nleft, size_t done>
    struct const_unroller {
        const UType &ref_;
        template<typename Functor>
        void for_each(const Functor &func) {
            func(ref_.arr_[COUNT - nleft]);
            const_unroller<nleft - 1, done + 1> ur(ref_);
            ur.for_each(func);
        }
        const_unroller(const UType &ref): ref_(ref) {}
    };
    template<size_t done>
    struct const_unroller<0, done> {
        template<typename Functor> void for_each(const Functor &func) {}
        const_unroller(const UType &ref) {}
    };
    template<typename Functor>
    void for_each(const Functor &func) {
        unroller<COUNT, 0> ur(*this);
        ur.for_each(func);
    }
    template<typename Functor>
    void for_each(const Functor &func) const {
        const_unroller<COUNT, 0> ur(*this);
        ur.for_each(func);
    }
    auto sum() const {
        ValueType ret = arr_[0];
        for(uint8_t i = 1; i < COUNT; ++i)
            ret += arr_[i];
        return ret;
    }
};

template<>
struct SIMDTypes<uint64_t> {
    using ValueType = uint64_t;
#if HAS_AVX_512
    using Type = __m512i;
    declare_all_int512(epi64, 512)
#elif __AVX2__
    using Type = __m256i;
    declare_all_int(epi64, 256)
#elif __SSE2__
    using Type = __m128i;
    declare_all_int128(epi64,)
#else
#error("Need at least sse2")
#endif
    static constexpr size_t ALN = sizeof(Type) / sizeof(char);
    static constexpr size_t MASK = ALN - 1;
    static constexpr size_t COUNT = sizeof(Type) / sizeof(ValueType);
    template<typename T>
    static constexpr bool aligned(T *ptr) {
        return (reinterpret_cast<uint64_t>(ptr) & MASK) == 0;
    }
    using VType = UType<SIMDTypes<ValueType>>;
};

template<>
struct SIMDTypes<float>{
    using ValueType = float;
#if HAS_AVX_512
    using Type = __m512;
    declare_all(ps, 512)
#ifndef NO_SLEEF
    dec_double_sz(__m512)
    dec_all_trig(f16, avx512f);
#endif // #ifndef NO_SLEEF
#elif __AVX2__
    using Type = __m256;
    declare_all(ps, 256)
#ifndef NO_SLEEF
    dec_double_sz(__m256)
    dec_all_trig(f8, avx2);
#endif // #ifndef NO_SLEEF
#elif __SSE2__
    using Type = __m128;
    declare_all(ps, )
#ifndef NO_SLEEF
    dec_double_sz(__m128)
    dec_all_trig(f4, sse2);
#endif // #ifndef NO_SLEEF
#else
#error("Need at least sse2")
#endif
    static constexpr size_t ALN = sizeof(Type) / sizeof(char);
    static constexpr size_t MASK = ALN - 1;
    static constexpr size_t COUNT = sizeof(Type) / sizeof(ValueType);
    template<typename T>
    static constexpr bool aligned(T *ptr) {
        return (reinterpret_cast<uint64_t>(ptr) & MASK) == 0;
    }
    using VType = UType<SIMDTypes<ValueType>>;
};

template<>
struct SIMDTypes<double>{
    using ValueType = double;
#if HAS_AVX_512
    using Type = __m512d;
    declare_all(pd, 512)
#ifndef NO_SLEEF
    dec_double_sz(__m512d);
    dec_all_trig(d8, avx512f);
#endif // #ifndef NO_SLEEF
#elif __AVX2__
    using Type = __m256d;
    declare_all(pd, 256)
#ifndef NO_SLEEF
    dec_double_sz(__m256d);
    dec_all_trig(d4, avx2);
#endif // #ifndef NO_SLEEF
#elif __SSE2__
    using Type = __m128d;
    declare_all(pd, )
#ifndef NO_SLEEF
    dec_double_sz(__m128d);
    dec_all_trig(d2, sse2);
#endif // #ifndef NO_SLEEF
#else
#error("Need at least sse2")
#endif
    static constexpr size_t ALN = sizeof(Type) / sizeof(char);
    static constexpr size_t MASK = ALN - 1;
    static constexpr size_t COUNT = sizeof(Type) / sizeof(ValueType);
    template<typename T>
    static constexpr bool aligned(T *ptr) {
        return (reinterpret_cast<uint64_t>(ptr) & MASK) == 0;
    }
    using VType = UType<SIMDTypes<ValueType>>;
};


template<typename FloatType>
void blockmul(FloatType *pos, size_t nelem, FloatType prod) {
#if __AVX2__ || HAS_AVX_512 || __SSE2__
        using SIMDType = typename SIMDTypes<FloatType>::Type;
        using Space = SIMDTypes<FloatType>;
        SIMDType factor(SIMDTypes<FloatType>::set1(prod));
        SIMDType *ptr(reinterpret_cast<SIMDType *>(pos));
        FloatType *end(pos + nelem);
        if(!Space::aligned(ptr)) {
            while(reinterpret_cast<FloatType *>(ptr) < end - sizeof(SIMDType) / sizeof(FloatType)) {
                Space::storeu(reinterpret_cast<FloatType *>(ptr),
                    Space::mul(factor, Space::loadu(reinterpret_cast<FloatType *>(ptr))));
                ++ptr;
            }
        } else {
            while(reinterpret_cast<FloatType *>(ptr) < end - sizeof(SIMDType) / sizeof(FloatType)) {
                Space::store(reinterpret_cast<FloatType *>(ptr),
                    Space::mul(factor, Space::load(reinterpret_cast<FloatType *>(ptr))));
                ++ptr;
            }
        }
        pos = reinterpret_cast<FloatType *>(ptr);
        while(pos < end) *pos++ *= prod;
#else
            for(size_t i(0); i < (static_cast<size_t>(1) << nelem); ++i) pos[i] *= prod; // Could be vectorized.
#endif
}

#define BLOCKOP(op, scalar) \
template<typename Container>\
void block##op(Container &con, double val) {\
    if(&con[1] - &con[0] == 1)\
        block##op(&con[0], con.size(), static_cast<std::decay_t<decltype(*std::begin(con))>>(val));\
    else\
        for(auto &el: con) scalar;\
}

template<typename FloatType>
void blockadd(FloatType *pos, size_t nelem, FloatType val) {
#if __AVX2__ || HAS_AVX_512 || __SSE2__
        using SIMDType = typename SIMDTypes<FloatType>::Type;
        using Space = SIMDTypes<FloatType>;
        SIMDType inc(SIMDTypes<FloatType>::set1(val));
        SIMDType *ptr(reinterpret_cast<SIMDType *>(pos));
        FloatType *end(pos + nelem);
        if(!Space::aligned(ptr))
            while(reinterpret_cast<FloatType *>(ptr) < end - sizeof(SIMDType) / sizeof(FloatType))
                Space::storeu(reinterpret_cast<FloatType *>(ptr),
                    Space::add(inc, Space::loadu(reinterpret_cast<FloatType *>(ptr)))), ++ptr;
        else
            while(reinterpret_cast<FloatType *>(ptr) < end - sizeof(SIMDType) / sizeof(FloatType))
                Space::store(reinterpret_cast<FloatType *>(ptr),
                    Space::add(inc, Space::load(reinterpret_cast<FloatType *>(ptr)))), ++ptr;
        pos = reinterpret_cast<FloatType *>(ptr);
        while(pos < end) *pos++ += val;
#else
            for(size_t i(0); i < (static_cast<size_t>(1) << nelem); ++i) pos[i] += val; // Could be vectorized.
#endif
}

BLOCKOP(mul, el *= val)
BLOCKOP(add, el += val)


#ifndef DO_DUFF
#define DO_DUFF(len, ITER) \
    do { \
        if(len) {\
            std::uint64_t loop = (len + 7) >> 3;\
            switch(len & 7) {\
                case 0: do {\
                    ITER; [[fallthrough]];\
                    case 7: ITER; [[fallthrough]]; case 6: ITER; [[fallthrough]]; case 5: ITER; [[fallthrough]];\
                    case 4: ITER; [[fallthrough]]; case 3: ITER; [[fallthrough]]; case 2: ITER; [[fallthrough]]; case 1: ITER;\
                } while (--loop);\
            }\
        }\
    } while(0)
#endif

template<typename FloatType>
void vecmul(FloatType *to, const FloatType *from, size_t nelem) {
#if __AVX2__ || HAS_AVX_512 || __SSE2__
        using SIMDType = typename SIMDTypes<FloatType>::Type;
        using Space = SIMDTypes<FloatType>;
        SIMDType *ptr(reinterpret_cast<SIMDType *>(to)), *fromptr(reinterpret_cast<SIMDType *>(from));
        FloatType *end(to + nelem);
        if(!(Space::aligned(ptr) && Space::aligned(fromptr)))
            while(reinterpret_cast<FloatType *>(ptr) < end - sizeof(SIMDType) / sizeof(FloatType))
                Space::storeu(reinterpret_cast<FloatType *>(ptr),
                    Space::mul(Space::loadu(reinterpret_cast<FloatType *>(fromptr)), Space::loadu(reinterpret_cast<FloatType *>(ptr)))),
                ++ptr, ++fromptr;
        else
            while(reinterpret_cast<FloatType *>(ptr) < end - sizeof(SIMDType) / sizeof(FloatType))
                Space::store(reinterpret_cast<FloatType *>(ptr),
                    Space::mul(Space::load(reinterpret_cast<FloatType *>(fromptr)), Space::load(reinterpret_cast<FloatType *>(ptr)))),
                ++ptr, ++fromptr;
        to = reinterpret_cast<FloatType *>(ptr), from = reinterpret_cast<FloatType *>(fromptr);
        while(to < end) *to++ *= *from++;
#else
        DO_DUFF(nelem, *to++ *= *from++);
#endif
}

template<typename FloatType, typename Functor>
void block_apply(FloatType *pos, size_t nelem, const Functor &func=Functor{}) {
#if __AVX2__ || HAS_AVX_512 || __SSE2__
        using Space = SIMDTypes<FloatType>;
        using SIMDType = typename Space::Type;
        SIMDType *ptr(reinterpret_cast<SIMDType *>(pos));
        FloatType *end(pos + nelem);
        if(!Space::aligned(ptr)) {
            while(reinterpret_cast<FloatType *>(ptr) < end - sizeof(SIMDType) / sizeof(FloatType)) {
                Space::storeu(reinterpret_cast<FloatType *>(ptr),
                    func(Space::loadu(reinterpret_cast<FloatType *>(ptr))));
                ++ptr;
            }
        } else {
            while(reinterpret_cast<FloatType *>(ptr) < end - sizeof(SIMDType) / sizeof(FloatType)) {
                Space::store(reinterpret_cast<FloatType *>(ptr),
                    func(Space::load(reinterpret_cast<FloatType *>(ptr))));
                ++ptr;
            }
        }
        pos = reinterpret_cast<FloatType *>(ptr);
        while(pos < end) *pos  = func.scalar(*pos), ++pos;
#else
            for(size_t i(0); i < (static_cast<size_t>(1) << nelem); ++i) to[i] *= func.scalar(to[i]); // Could be vectorized.
#endif
}

template<typename Container, typename Functor>
void block_apply(Container &con, const Functor &func=Functor{}) {
#ifndef NO_BLAZE
    if constexpr(IS_CONTIGUOUS_UNCOMPRESSED_BLAZE(Container)) {
        const size_t nelem(con.size());
        block_apply(&(*std::begin(con)), nelem, func);
    } else {
#endif
        if(&con[1] - &con[0] == 1) {
        const size_t nelem(con.size());
        block_apply(&(*std::begin(con)), nelem, func);
        } else for(auto &el: con) el = func.scalar(el);
#ifndef NO_BLAZE
    }
#endif
}

template<typename T, typename SizeType=std::size_t>
void memblockset(void *dest, T val, SizeType nbytes) {
    using S = SIMDTypes<uint64_t>;
    using SType = typename S::Type;
    SType sv;
    {
        T *s(reinterpret_cast<T *>(&sv)), *s2(reinterpret_cast<T *>(((reinterpret_cast<char *>(&sv)) + sizeof(sv))));
        while(s < s2) *s++ = val;
    }
    if(S::aligned(dest)) for(SType *s = static_cast<SType *>(dest), *e = reinterpret_cast<SType *>(static_cast<char *>(dest) + nbytes); s < e; *s++ = sv);
    else for(SType *s = static_cast<SType *>(dest), *e = reinterpret_cast<SType *>(static_cast<char *>(dest) + nbytes); s < e; S::storeu(s++, sv));
}

} // namespace vec
#ifndef NO_SLEEF
#undef OP
#undef SLEEF_OP
#undef dec_sleefop_prec
#undef dec_all_precs
#undef dec_all_precs_u05
#undef dec_all_trig
#undef dec_double_sz
#endif // #ifndef NO_SLEEF

#undef declare_all
#undef decop

#endif // #ifndef _VEC_H__
