#include "sse_v1.h"

#include <algorithm>
#include <immintrin.h>

int64_t CountTri_N2_LinearSearchSSE(AlignedIntArray edges)
{
    std::sort(edges.begin(), edges.end());
    size_t N = edges.size();

    int64_t total = 0;

#ifdef USE_OPENMP
    #pragma omp parallel for
#endif
    for (size_t i_largest = 2; i_largest < N; ++i_largest) {

        ssize_t i_lower = 0;
        ssize_t i_upper = i_largest - 1;

        if (edges[i_upper - 1] + edges[i_upper] <= edges[i_largest]) continue;

        __m128i largest_reg = _mm_set1_epi32(edges[i_largest] + 1);
        __m128i zero_reg  = _mm_setzero_si128();
        int32_t unroll = 4;
        __m128i n_unroll_reg = _mm_set1_epi32(-unroll);
        __m128i num_tri0 = _mm_setzero_si128();
        __m128i num_tri1 = _mm_setzero_si128();

        while (i_lower < i_upper) {
            // i_lower = LinearSearch(edges, i_lower, i_upper, edges[i_largest] - edges[i_upper]);
            // num_tri += i_upper - i_lower;
            // --i_upper;
            __m128i i_lowers_reg = _mm_set1_epi32(i_lower);
            __m128i i_uppers_reg = _mm_setr_epi32(i_upper - 3, i_upper - 2, i_upper - 1, i_upper - 0);
            __m128i uppers_reg   = _mm_loadu_si128((__m128i *)&edges[i_upper - 3]);

            __m128i ref_reg = _mm_sub_epi32(largest_reg, uppers_reg);

            __m128i lower0 = _mm_set1_epi32(edges[i_lower + 0]);
            __m128i lower1 = _mm_set1_epi32(edges[i_lower + 1]);
            __m128i lower2 = _mm_set1_epi32(edges[i_lower + 2]);
            __m128i lower3 = _mm_set1_epi32(edges[i_lower + 3]);

            __m128i is0  = _mm_cmplt_epi32(lower0, ref_reg);
            __m128i is1  = _mm_cmplt_epi32(lower1, ref_reg);
            __m128i is2  = _mm_cmplt_epi32(lower2, ref_reg);
            __m128i is3  = _mm_cmplt_epi32(lower3, ref_reg);

            __m128i is01 = _mm_add_epi32(is0, is1);
            __m128i is23 = _mm_add_epi32(is2, is3);

            __m128i sum  = _mm_add_epi32(is01, is23);

            auto mask = _mm_movemask_epi8(_mm_cmpeq_epi32(sum, n_unroll_reg));
            if (mask) {  // fallback
                for (size_t i = i_lower + unroll; i < i_upper; i += unroll) {
                    __m128i lower0 = _mm_set1_epi32(edges[i + 0]);
                    __m128i lower1 = _mm_set1_epi32(edges[i + 1]);
                    __m128i lower2 = _mm_set1_epi32(edges[i + 2]);
                    __m128i lower3 = _mm_set1_epi32(edges[i + 3]);

                    __m128i is0  = _mm_cmplt_epi32(lower0, ref_reg);
                    __m128i is1  = _mm_cmplt_epi32(lower1, ref_reg);
                    __m128i is2  = _mm_cmplt_epi32(lower2, ref_reg);
                    __m128i is3  = _mm_cmplt_epi32(lower3, ref_reg);
                    __m128i is01 = _mm_add_epi32(is0, is1);
                    __m128i is23 = _mm_add_epi32(is2, is3);
                    __m128i is0123  = _mm_add_epi32(is01, is23);
                    sum = _mm_add_epi32(sum, is0123);
                    auto mask = _mm_movemask_epi8(_mm_cmpeq_epi32(is0123, n_unroll_reg));
                    if (!mask) break;
                }
            }

            __m128i new_i_lowers = _mm_sub_epi32(zero_reg, sum);
            new_i_lowers = _mm_add_epi32(new_i_lowers, i_lowers_reg);
            new_i_lowers = _mm_min_epi32(new_i_lowers, i_uppers_reg);

            i_lower = _mm_extract_epi32(new_i_lowers, 0);

            __m128i new_tris = _mm_sub_epi32(i_uppers_reg, new_i_lowers);

            __m128i a = _mm_cvtepu32_epi64(new_tris);
            __m128i b = _mm_cvtepu32_epi64(_mm_shuffle_epi32(new_tris, _MM_SHUFFLE(0, 1, 2, 3)));
            num_tri0 = _mm_add_epi64(num_tri0, a);
            num_tri1 = _mm_add_epi64(num_tri1, b);

            i_upper -= 4;
        }

        int64_t vals[4] = {};
        _mm_storeu_si128((__m128i *)&vals[0], num_tri0);
        _mm_storeu_si128((__m128i *)&vals[2], num_tri1);
#ifdef USE_OPENMP
        #pragma omp atomic
#endif
        total += vals[0] + vals[1] + vals[2] + vals[3];
    }

    return total;
}
