#include <mitsuba/core/profiler.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/render/sampler.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class FixedSampler final : public PCG32Sampler<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(PCG32Sampler, m_sample_count, m_base_seed, m_rng, seed, seeded)
    MTS_IMPORT_TYPES()

    FixedSampler(const Properties &props = Properties()) : Base(props) {
        /* Can't seed yet on the GPU because we don't know yet
           how many entries will be needed. */
        if (!is_dynamic_array_v<Float>)
            seed(PCG32_DEFAULT_STATE);
    }

    ref<Sampler<Float, Spectrum>> clone() override {
        FixedSampler *sampler = new FixedSampler();
        sampler->m_sample_count = m_sample_count;
        sampler->m_base_seed = m_base_seed;
        return sampler;
    }

    Float next_1d(Mask active = true) override {
        Assert(seeded());
        // return m_rng.template next_float<Float>(active);
        return 0.5f;
    }

    Point2f next_2d(Mask active = true) override {
        Float f1 = next_1d(active),
              f2 = next_1d(active);
        // return Point2f(f1, f2);
        return Point2f(0.5f, 0.5f);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "FixedSampler[" << std::endl
            << "  sample_count = " << m_sample_count << std::endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(FixedSampler, Sampler)
MTS_EXPORT_PLUGIN(FixedSampler, "Fixed Sampler");
NAMESPACE_END(mitsuba)
