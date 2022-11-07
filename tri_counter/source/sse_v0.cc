#include "sse_v0.h"

#include <algorithm>
#include <immintrin.h>

static ssize_t LinearSearchScalar(const AlignedIntArray &arr, size_t i_start, size_t i_end, int ref)
{
    for (size_t i = i_start; i < i_end; i += 1) {
        if (arr[i + 0] > ref) return i;
    }
    return i_end;
}

static ssize_t LinearSearchScalarUnrolled(const AlignedIntArray &arr, size_t i_start, size_t i_end, int ref)
{
    for (size_t i = i_start; i < i_end; i += 4) {

        bool is0 = arr[i + 0] > ref;
        bool is1 = arr[i + 1] > ref;
        bool is2 = arr[i + 2] > ref;
        bool is3 = arr[i + 3] > ref;

        if (is0 | is1 | is2 | is3) {
            if (is0) return i + 0;
            if (is1) return i + 1;
            if (is2) return i + 2;
            if (is3) return i + 3;
        }
    }
    return -1;
}


static ssize_t LinearSearchSSE4(const AlignedIntArray &arr, size_t i_start, size_t i_end, int ref)
{
    const __m128i *p = arr.m128i();
    __m128i ref_reg = _mm_set1_epi32(ref + 1);

    for (size_t i = i_start / 4; i < i_end / 4; i += 4) {
        // "<=x" <=> "<x+1"
        __m128i is0    = _mm_cmplt_epi32(p[i + 0], ref_reg);
        __m128i is1    = _mm_cmplt_epi32(p[i + 1], ref_reg);
        __m128i is2    = _mm_cmplt_epi32(p[i + 2], ref_reg);
        __m128i is3    = _mm_cmplt_epi32(p[i + 3], ref_reg);
        __m128i is01   = _mm_packs_epi32(is0, is1);
        __m128i is23   = _mm_packs_epi32(is2, is3);
        __m128i is0123 = _mm_packs_epi16(is01, is23);

        auto mask = _mm_movemask_epi8(is0123);
        if (mask != 0xFFFF) return i*4 + __builtin_ctz(~mask);
    }

    return -1;
}

static ssize_t LinearSearchSSE1(const AlignedIntArray &arr, size_t i_start, size_t i_end, int ref)
{
    const __m128i *p = arr.m128i();
    __m128i ref_reg = _mm_set1_epi32(ref + 1);

    for (size_t i = i_start / 4; i < i_end / 4; i += 1) {
        __m128i is0 = _mm_cmplt_epi32(p[i], ref_reg);  // "<=x" <=> "<x+1"
        auto mask = _mm_movemask_epi8(is0);
        if (mask != 0xFFFF) return i*4 + __builtin_ctz(~mask) / 4;
    }

    return -1;
}

static ssize_t LinearSearchSSE_Branchless(const AlignedIntArray &arr, size_t i_start, size_t i_end, int ref)
{
    const __m128i *p = arr.m128i();
    __m128i ref_reg = _mm_set1_epi32(ref + 1);

    __m128i counter = _mm_setzero_si128();
    for (int i = i_start / 4; i < i_end / 4; i += 4) {
        __m128i is0  = _mm_cmplt_epi32(p[i + 0], ref_reg);
        __m128i is1  = _mm_cmplt_epi32(p[i + 1], ref_reg);
        __m128i is2  = _mm_cmplt_epi32(p[i + 2], ref_reg);
        __m128i is3  = _mm_cmplt_epi32(p[i + 3], ref_reg);
        __m128i is01 = _mm_add_epi32(is0, is1);
        __m128i is23 = _mm_add_epi32(is2, is3);
        __m128i sum  = _mm_add_epi32(is01, is23);

        counter = _mm_sub_epi32(counter, sum);
        auto mask = _mm_movemask_epi8(_mm_cmpeq_epi32(is3, _mm_set1_epi32(0)));
        if (mask == 0xFFFF) break;
    }
    counter = _mm_add_epi32(counter, _mm_shuffle_epi32(counter, _MM_SHUFFLE(2, 3, 0, 1)));
    counter = _mm_add_epi32(counter, _mm_shuffle_epi32(counter, _MM_SHUFFLE(1, 0, 3, 2)));
    // a[0] = a[0] + a[2] + a[1] + a[3]
    // a[1] = a[1] + a[3] + a[0] + a[2]
    // a[2] = a[2] + a[0] + a[3] + a[1]
    // a[3] = a[3] + a[1] + a[2] + a[0]
    ssize_t result = _mm_cvtsi128_si32(counter) + i_start;
    return result < i_end ? result : -1;
}


template <typename SIMDSearchFunc, size_t IndexAlignment = 16>
size_t LinearSearchSIMD(const AlignedIntArray &arr, size_t i_lower, size_t i_upper, int ref, const SIMDSearchFunc &f)
{
    size_t mask = IndexAlignment - 1;
    size_t i_upper4 = i_upper & ~mask;
    size_t i_lower4 = (i_lower + mask) & ~mask;
    if (i_lower4 >= i_upper4) return LinearSearchScalar(arr, i_lower, i_upper, ref);

    for (size_t i = i_lower; i < i_lower4; ++i) {
        if (arr[i] > ref) return i;
    }

    ssize_t i_found = f(arr, i_lower4, i_upper4, ref);
    if (i_found >= 0) return i_found;

    for (size_t i = i_upper4; i < i_upper; ++i) {
        if (arr[i] > ref) return i;
    }
    return i_upper;
}


template<typename LinearSearchSIMDFunc>
size_t CountTri_SIMD1(AlignedIntArray edges, const LinearSearchSIMDFunc &f)
{
    std::sort(edges.begin(), edges.end());
    size_t N = edges.size();
    int64_t num_tri = 0;
    for (size_t i_largest = 2; i_largest < N; ++i_largest) {

        size_t i_lower = 0;
        size_t i_upper = i_largest - 1;

        if (edges[i_upper - 1] + edges[i_upper] <= edges[i_largest]) continue;

        while (i_lower < i_upper) {
            i_lower = LinearSearchSIMD(edges, i_lower, i_upper, edges[i_largest] - edges[i_upper], f);
            num_tri += i_upper - i_lower;
            --i_upper;
        }
    }
    return num_tri;
}

size_t CountTri_SSE4(const AlignedIntArray &edges)
{
    return CountTri_SIMD1(edges, LinearSearchSSE4);
}

size_t CountTri_SSE1(const AlignedIntArray &edges)
{
    return CountTri_SIMD1(edges, LinearSearchSSE1);
}

size_t CountTri_SSE_Branchless(const AlignedIntArray &edges)
{
    return CountTri_SIMD1(edges, LinearSearchSSE_Branchless);
}

size_t CountTri_ScalarUR(const AlignedIntArray &edges)
{
    return CountTri_SIMD1(edges, LinearSearchScalarUnrolled);
}
