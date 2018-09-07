#ifndef WELFORD_ONLINE_STDEV_H__
#define WELFORD_ONLINE_STDEV_H__

// based on John D. Cook's blog https://www.johndcook.com/blog/standard_deviation/
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "blaze/Math.h"

namespace stats {

template<typename T, typename SizeType=std::uint64_t,
         typename=typename std::enable_if<std::is_floating_point<T>::value>::type,
         typename=typename std::enable_if<std::is_unsigned<SizeType>::value>::type>
class OnlineSD {
    T old_mean_, new_mean_, olds_, news_;
    SizeType n_;
public:
    OnlineSD() {std::memset(this, 0, sizeof(*this));}

    void add(T x)
    {
        // See Knuth TAOCP vol 2, 3rd edition, page 232
        if (__builtin_expect(++n_ == 1, 0)) old_mean_ = new_mean_ = x, olds_ = 0.;
        else
        {
            new_mean_ = old_mean_ + (x - old_mean_)/n_;
            news_ = olds_ + (x - old_mean_)*(x - new_mean_);
            // set up for next iteration
            old_mean_ = new_mean_, olds_ = news_;
        }
    }
    size_t n()   const {return n_;}
    T mean()     const {return n_ ? new_mean_: 0.0;}
    T variance() const {return n_ > 1 ? news_ / (n_ - 1): 0.0;}
    T stdev()    const {return std::sqrt(variance());}
};

template<typename VecType=::blaze::DynamicVector<double, ::blaze::rowVector>, typename SizeType=std::uint64_t>
class OnlineVectorSD {
    VecType old_mean_, new_mean_, olds_, news_;
    SizeType n_;
public:

    template<typename VType2>
    OnlineVectorSD(const VType2 &vec): old_mean_(vec.size()), new_mean_(vec.size()), olds_(vec.size()), news_(vec.size()), n_(0) {
        old_mean_.reset();
        olds_.reset();
        new_mean_.reset();
        news_.reset();
    }

    template<typename VType2>
    void add(const VType2 &x)
    {
        // See Knuth TAOCP vol 2, 3rd edition, page 232
        if (__builtin_expect(++n_ == 1, 0)) old_mean_ = new_mean_ = x, olds_ = 0.;
        else
        {
            new_mean_ = old_mean_ + (x - old_mean_)* (1./n_);
            news_ = olds_ + (x - old_mean_)*(x - new_mean_);
            // set up for next iteration
            old_mean_ = new_mean_, olds_ = news_;
        }
    }
#define ASSERTFULL() do {if(!n_) {throw std::runtime_error("Cannot calculate stats on an empty stream.");}} while(0)
    size_t n()   const {return n_;}
    VecType mean()     const {ASSERTFULL(); return new_mean_;}
    VecType variance() const {ASSERTFULL(); return news_ / (n_ - 1);}
    VecType stdev()    const {ASSERTFULL(); return blaze::sqrt(variance());}
};

template<typename T=float, typename SizeType=std::int64_t,
         typename=typename std::enable_if<std::is_floating_point<T>::value>::type,
         typename=typename std::enable_if<std::is_integral<SizeType>::value>::type>
class OnlineStatistics
{
public:
    OnlineStatistics() {clear();}
    void clear() {std::memset(this, 0, sizeof(*this));}
    void add(T x) {
        T delta, delta_n, delta_n2, term1;

        SizeType n1 = n_++;
        delta = x - m1_;
        delta_n = delta / n_;
        delta_n2 = delta_n * delta_n;
        term1 = delta * delta_n * n1;
        m1_ += delta_n;
        m4_ += term1 * delta_n2 * (n_*n_ - 3*n_ + static_cast<SizeType>(3)) + \
              6. * delta_n2 * m2_ - 4. * delta_n * m3_;
        m3_ += term1 * delta_n * (n_ - 2) - 3 * delta_n * m2_;
        m2_ += term1;
    }
    SizeType n() const {return n_;}
    T mean() const {return m1_;}
    T variance() const {return m2_/(n_- (n_ > 1.0));}
    T stdev() const    {return std::sqrt(variance());}
    T skewness() const {
        assert(m2_ >= 0.);
        return std::sqrt(static_cast<T>(n_)) * m3_/ std::pow(m2_, 1.5);
    }
    T kurtosis() const {return static_cast<T>(n_)*m4_ / (m2_*m2_) - 3.0;}

    friend OnlineStatistics operator+(const OnlineStatistics a, const OnlineStatistics b);
    OnlineStatistics& operator+=(const OnlineStatistics& rhs)
    {
            OnlineStatistics combined = *this + rhs;
            return *this = combined;
    }

private:
    T m1_, m2_, m3_, m4_;
    SizeType n_;
};

template<typename T=float, typename SizeType=std::int32_t,
         typename=typename std::enable_if<std::is_floating_point<T>::value>::type,
         typename=typename std::enable_if<std::is_integral<SizeType>::value>::type>
OnlineStatistics<T, SizeType> operator+(const OnlineStatistics<T, SizeType> &a,
                                        const OnlineStatistics<T, SizeType> &b) {
    OnlineStatistics<T, SizeType> combined;

    combined.n = a.n + b.n;

    const T delta = b.m1_ - a.m1_;
    const T delta2 = delta*delta;
    const T delta3 = delta*delta2;
    const T delta4 = delta2*delta2;

    combined.m1_ = (a.n*a.m1_ + b.n*b.m1_) / combined.n;

    combined.m2_ = a.m2_ + b.m2_ +
                  delta2 * a.n * b.n / combined.n;

    combined.m3_ = a.m3_ + b.m3_ +
                  delta3 * a.n * b.n * (a.n - b.n)/(combined.n*combined.n);
    combined.m3_ += 3.0*delta * (a.n*b.m2_ - b.n*a.m2_) / combined.n;

    combined.m4_ = a.m4_ + b.m4_ + delta4*a.n*b.n * (a.n*a.n - a.n*b.n + b.n*b.n) /
                  (combined.n*combined.n*combined.n);
    combined.m4_ += 6.0*delta2 * (a.n*a.n*b.m2_ + b.n*b.n*a.m2_)/(combined.n*combined.n) +
                  4.0*delta*(a.n*b.m3_ - b.n*a.m3_) / combined.n;

    return combined;
}

} // namespace stats

#endif /* #ifndef WELFORD_ONLINE_STDEV_H__ */
