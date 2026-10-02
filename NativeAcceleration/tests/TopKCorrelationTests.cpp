// GPU-free check of the insertion routine used by UpdateCorrelationTopKKernel.
// clang++ -std=c++14 -O2 NativeAcceleration/tests/TopKCorrelationTests.cpp -o /tmp/warp-topk-test && /tmp/warp-topk-test
#include "../gtom/include/TopKCorrelation.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

int main()
{
    std::mt19937 random(813);
    constexpr size_t voxels = 7;
    const float infinity = std::numeric_limits<float>::infinity();

    for (unsigned int count : {0U, 1U, 7U, 8U, 9U, 23U, 53U})
        for (unsigned int topk : {1U, 2U, 8U, 64U})
            for (unsigned int batchsize : {1U, 4U, 8U, 32U})
                for (bool shuffled : {false, true})
                {
                    std::vector<float> source(count * voxels);
                    for (float& value : source)
                        value = (int)(random() % 13) - 8; // Deliberate ties, including negative scores.
                    if (count > 7)
                    {
                        source[1] = std::numeric_limits<float>::quiet_NaN();
                        source[voxels + 2] = infinity;
                        source[voxels * 2 + 3] = -infinity;
                    }
                    std::vector<unsigned int> order(count);
                    std::iota(order.begin(), order.end(), 0);
                    if (shuffled)
                        std::shuffle(order.begin(), order.end(), random);

                    std::vector<float> scores(topk * voxels, -infinity);
                    std::vector<float> angles(topk * voxels, -1);
                    for (unsigned int batchoffset = 0; batchoffset < count; batchoffset += batchsize)
                    {
                        const unsigned int current = std::min(batchsize, count - batchoffset);
                        // Poison unused tail slots: accidentally visiting one would beat every valid score.
                        std::vector<float> batch(batchsize * voxels, 1000000);
                        for (unsigned int b = 0; b < current; ++b)
                            std::copy_n(source.data() + order[batchoffset + b] * voxels,
                                        voxels, batch.data() + b * voxels);
                        for (size_t voxel = 0; voxel < voxels; ++voxel)
                            for (unsigned int b = 0; b < current; ++b)
                                gtom::InsertCorrelationTopK(batch[b * voxels + voxel],
                                    (float)order[batchoffset + b], scores.data(), angles.data(), voxel, voxels, topk);
                    }

                    for (size_t voxel = 0; voxel < voxels; ++voxel)
                    {
                        std::vector<std::pair<float, unsigned int>> expected;
                        for (unsigned int angle = 0; angle < count; ++angle)
                            if (std::isfinite(source[angle * voxels + voxel]))
                                expected.emplace_back(source[angle * voxels + voxel], angle);
                        std::sort(expected.begin(), expected.end(), [](const auto& a, const auto& b)
                        {
                            return a.first != b.first ? a.first > b.first : a.second < b.second;
                        });
                        for (unsigned int rank = 0; rank < topk; ++rank)
                        {
                            const size_t offset = rank * voxels + voxel;
                            assert(scores[offset] == (rank < expected.size() ? expected[rank].first : -infinity));
                            assert(angles[offset] == (rank < expected.size() ? expected[rank].second : -1.0f));
                        }
                    }
                }

    float scores[2] = {-infinity, -infinity};
    float angles[2] = {-1, -1};
    gtom::InsertCorrelationTopK(3.0f, 16777215.0f, scores, angles, 0, 1, 2);
    gtom::InsertCorrelationTopK(3.0f, 16777214.0f, scores, angles, 0, 1, 2);
    assert(angles[0] == 16777214.0f && angles[1] == 16777215.0f);
    std::cout << "Top-K insertion matches exhaustive sorting across ties, orderings, K, and partial batches.\n";
}
