#pragma once

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <limits>
#include <vector>

#include <immintrin.h>

struct AlignedIntArray {
    static constexpr size_t  kPadding           = 32;
    static constexpr int32_t kUpperPaddingValue = std::numeric_limits<int32_t>::max();
    static constexpr int32_t kLowerPaddingValue = std::numeric_limits<int32_t>::min();

    AlignedIntArray(const std::vector<int> &data, int alignment = 4048)
    {
        m_alignment = alignment;
        m_num_elements = data.size();
        size_t size = m_num_elements * sizeof(*m_data);
        size_t padding_size = kPadding * sizeof(*m_data);
        m_padded_data = (int *)aligned_alloc(m_alignment, size + 2 * padding_size);
        m_data = m_padded_data + kPadding;
        memmove(m_data, data.data(), size);
        for(size_t i = m_num_elements; i < m_num_elements + kPadding; ++i) {
            m_data[i] = kUpperPaddingValue;
        }
        for(size_t i = 0; i < kPadding; ++i) {
            m_padded_data[i] = kLowerPaddingValue;
        }
    }

    AlignedIntArray(const AlignedIntArray &data) noexcept
    {
        m_num_elements = data.size();
        m_alignment = data.alignment();
        size_t size = m_num_elements * sizeof(*m_data);
        size_t padding_size = kPadding * sizeof(*m_data);
        m_padded_data = (int *)aligned_alloc(m_alignment, size + 2 * padding_size);
        m_data = m_padded_data + kPadding;
        memmove(m_data, data.data(), size);
        for(size_t i = m_num_elements; i < m_num_elements + kPadding; ++i) {
            m_data[i] = kUpperPaddingValue;
        }
        for(size_t i = 0; i < kPadding; ++i) {
            m_padded_data[i] = kLowerPaddingValue;
        }
    }

    ~AlignedIntArray()
    {
        free(m_padded_data);
        m_data = nullptr;
        m_padded_data = nullptr;
        m_num_elements = 0;
    }

    int *begin()
    {
        return &m_data[0];
    }

    const int *begin() const
    {
        return &m_data[0];
    }

    int *end()
    {
        return &m_data[m_num_elements - 1];
    }

    const int *end() const
    {
        return &m_data[m_num_elements - 1];
    }

    int &operator[](size_t idx)
    {
        return m_data[idx];
    }

    const int &operator[](size_t idx) const
    {
        return m_data[idx];
    }

    constexpr size_t size() const
    {
        return m_num_elements;
    }

    constexpr int *data() const
    {
        return m_data;
    }

    constexpr int alignment() const
    {
        return m_alignment;
    }

    const __m128i *m128i() const
    {
        return (const __m128i *)m_data;
    }

  private:
    int     m_alignment;
    int    *m_data;
    int    *m_padded_data;
    size_t  m_num_elements;
};


struct Profile {
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;

    Profile()
    {
        m_start_time = clock::now();
    }

    ~Profile()
    {
        using std::chrono::duration_cast;
        using std::chrono::microseconds;

        auto total_us = duration_cast<microseconds>(clock::now() - m_start_time).count();

        int us = total_us % 1000;
        int ms = total_us / 1000 % 1000;
        int s  = total_us / 1000 / 1000 % 60;
        int m  = total_us / 1000 / 1000 / 60;

        printf("Time: %d m. %d s. %d ms. %d us.\n", m, s, ms, us);
    }

    time_point  m_start_time;
};
