#include <random>

#include <enoki/stl.h>
#include <enoki/morton.h>

#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/imageblock.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/spiral.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class ReorderIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sensor, Sampler, ImageBlock, Film, Medium, Emitter,
                     EmitterPtr, BSDF, BSDFPtr)

    ReorderIntegrator(const Properties &props) : Base(props) { }

    bool render(Scene *scene, Sensor *sensor) override {
        ScopedPhase sp(ProfilerPhase::Render);
        this->m_stop = false;

        ref<Film> film           = sensor->film();
        ScalarVector2i film_size = film->crop_size();

        size_t total_spp = sensor->sampler()->sample_count();
        size_t samples_per_pass =
            (this->m_samples_per_pass == (size_t) -1)
                ? total_spp
                : std::min((size_t) this->m_samples_per_pass, total_spp);
        if ((total_spp % samples_per_pass) != 0)
            Throw("sample_count (%d) must be a multiple of samples_per_pass "
                  "(%d).",
                  total_spp, samples_per_pass);

        size_t n_passes = (total_spp + samples_per_pass - 1) / samples_per_pass;

        std::vector<std::string> channels = this->aov_names();
        bool has_aovs                     = !channels.empty();

        // Insert default channels and set up the film
        for (size_t i = 0; i < 5; ++i)
            channels.insert(channels.begin() + i, std::string(1, "XYZAW"[i]));
        film->prepare(channels);

        this->m_render_timer.reset();
        if constexpr (!is_cuda_array_v<Float>) {
            /// Render on the CPU using a spiral pattern
            size_t n_threads = __global_thread_count;
            Log(Info, "Starting render job (%ix%i, %i sample%s,%s %i thread%s)",
                film_size.x(), film_size.y(), total_spp,
                total_spp == 1 ? "" : "s",
                n_passes > 1 ? tfm::format(" %d passes,", n_passes) : "",
                n_threads, n_threads == 1 ? "" : "s");

            if (this->m_timeout > 0.f)
                Log(Info, "Timeout specified: %.2f seconds.", this->m_timeout);

            // Find a good block size to use for splitting up the total
            // workload.
            if (this->m_block_size == 0) {
                uint32_t block_size = MTS_BLOCK_SIZE;
                while (true) {
                    if (block_size == 1 || hprod((film_size + block_size - 1) /
                                                 block_size) >= n_threads)
                        break;
                    block_size /= 2;
                }
                this->m_block_size = block_size;
            }

            Spiral spiral(film, this->m_block_size, n_passes);

            ThreadEnvironment env;
            ref<ProgressReporter> progress = new ProgressReporter("Rendering");
            std::mutex mutex;

            // Total number of blocks to be handled, including multiple passes.
            size_t total_blocks = spiral.block_count() * n_passes,
                   blocks_done  = 0;

            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, total_blocks, 1),
                [&](const tbb::blocked_range<size_t> &range) {
                    ScopedSetThreadEnvironment set_env(env);
                    ref<Sampler> sampler  = sensor->sampler()->clone();
                    ref<ImageBlock> block = new ImageBlock(
                        this->m_block_size, channels.size(),
                        film->reconstruction_filter(), !has_aovs);
                    scoped_flush_denormals flush_denormals(true);
                    std::unique_ptr<Float[]> aovs(new Float[channels.size()]);

                    // For each block
                    for (auto i = range.begin();
                         i != range.end() && !this->should_stop(); ++i) {
                        auto [offset, size, block_id] = spiral.next_block();
                        Assert(hprod(size) != 0);
                        block->set_size(size);
                        block->set_offset(offset);

                        this->render_block(scene, sensor, sampler, block,
                                           aovs.get(), samples_per_pass,
                                           block_id);

                        film->put(block);

                        /* Critical section: update progress bar */ {
                            std::lock_guard<std::mutex> lock(mutex);
                            blocks_done++;
                            progress->update(blocks_done /
                                             (ScalarFloat) total_blocks);
                        }
                    }
                });
        }

        if (!this->m_stop)
            Log(Info, "Rendering finished. (took %s)",
                util::time_string(this->m_render_timer.value(), true));

        return !this->m_stop;
    }

    void render_block(const Scene *scene, const Sensor *sensor,
                      Sampler *sampler, ImageBlock *block, Float *aovs,
                      size_t sample_count_, size_t block_id) const override {
        block->clear();
        uint32_t pixel_count =
                     (uint32_t)(this->m_block_size * this->m_block_size),
                 sample_count = (uint32_t)(sample_count_ == (size_t) -1
                                               ? sampler->sample_count()
                                               : sample_count_);

        ScalarFloat diff_scale_factor =
            rsqrt((ScalarFloat) sampler->sample_count());

        if constexpr (!is_array_v<Float>) {
            for (uint32_t i = 0; i < pixel_count && !this->should_stop(); ++i) {
                sampler->seed(block_id * pixel_count + i);

                ScalarPoint2u pos = enoki::morton_decode<ScalarPoint2u>(i);
                if (any(pos >= block->size()))
                    continue;

                pos += block->offset();
                for (uint32_t j = 0; j < sample_count && !this->should_stop();
                     ++j) {
                    this->render_sample(scene, sensor, sampler, block, aovs,
                                        pos, diff_scale_factor);
                }
            }
        } else if constexpr (is_array_v<Float> && !is_cuda_array_v<Float>) {
            // Ensure that the sample generation is fully deterministic
            sampler->seed(block_id);

            for (auto [index, active] :
                 range<UInt32>(pixel_count * sample_count)) {
                if (this->should_stop())
                    break;
                Point2u pos =
                    enoki::morton_decode<Point2u>(index / UInt32(sample_count));
                active &= !any(pos >= block->size());
                pos += block->offset();
                this->render_sample(scene, sensor, sampler, block, aovs, pos,
                                    diff_scale_factor, active);
            }
        }
    }

    void render_sample(const Scene *scene, const Sensor *sensor,
                       Sampler *sampler, ImageBlock *block, Float *aovs,
                       const Vector2f &pos, ScalarFloat diff_scale_factor,
                       Mask active = true) const {
        Vector2f position_sample = pos + sampler->next_2d(active);

        Point2f aperture_sample(.5f);
        if (sensor->needs_aperture_sample())
            aperture_sample = sampler->next_2d(active);

        Float time = sensor->shutter_open();
        if (sensor->shutter_open_time() > 0.f)
            time += sampler->next_1d(active) * sensor->shutter_open_time();

        Float wavelength_sample = sampler->next_1d(active);

        Vector2f adjusted_position =
            (position_sample - sensor->film()->crop_offset()) /
            sensor->film()->crop_size();

        auto [ray, ray_weight] = sensor->sample_ray_differential(
            time, wavelength_sample, adjusted_position, aperture_sample);

        ray.scale_differential(diff_scale_factor);

        const Medium *medium = sensor->medium();
        std::pair<Spectrum, Mask> result =
            sample(scene, sampler, ray, medium, aovs + 5, active);
        result.first = ray_weight * result.first;

        UnpolarizedSpectrum spec_u = depolarize(result.first);

        Color3f xyz;
        if constexpr (is_monochromatic_v<Spectrum>) {
            xyz = spec_u.x();
        } else if constexpr (is_rgb_v<Spectrum>) {
            xyz = srgb_to_xyz(spec_u, active);
        } else {
            static_assert(is_spectral_v<Spectrum>);
            xyz = spectrum_to_xyz(spec_u, ray.wavelengths, active);
        }

        aovs[0] = xyz.x();
        aovs[1] = xyz.y();
        aovs[2] = xyz.z();
        aovs[3] = select(result.second, Float(1.f), Float(0.f));
        aovs[4] = 1.f;

        block->put(position_sample, aovs, active);

        sampler->advance();
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene,
                                     Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        RayDifferential3f ray = ray_;

        // Tracks radiance scaling due to index of refraction changes
        Float eta(1.f);

        // MIS weight for intersected emitters (set by prev. iteration)
        Float emission_weight(1.f);

        Spectrum throughput(1.f), result(0.f);

        // ---------------------- First intersection ----------------------

        SurfaceInteraction3f si = scene->ray_intersect(ray, active);
        Mask valid_ray = si.is_valid();
        EmitterPtr emitter = si.emitter(scene);

        for (int depth = 1;; ++depth) {

            // ---------------- Intersection with emitters ----------------

            if (any_or<true>(neq(emitter, nullptr)))
                result[active] += emission_weight * throughput * emitter->eval(si, active);

            active &= si.is_valid();

            /* Russian roulette: try to keep path weights equal to one,
               while accounting for the solid angle compression at refractive
               index boundaries. Stop with at least some probability to avoid
               getting stuck (e.g. due to total internal reflection) */
            if (depth > m_rr_depth) {
                Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
                active &= sampler->next_1d(active) < q;
                throughput *= rcp(q);
            }

            // Stop if we've exceeded the number of requested bounces, or
            // if there are no more active lanes. Only do this latter check
            // in GPU mode when the number of requested bounces is infinite
            // since it causes a costly synchronization.
            if ((uint32_t) depth >= (uint32_t) m_max_depth ||
                ((!is_cuda_array_v<Float> || m_max_depth < 0) && none(active)))
                break;

            // --------------------- Emitter sampling ---------------------

            BSDFContext ctx;
            BSDFPtr bsdf = si.bsdf(ray);
            Mask active_e = active && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            if (likely(any_or<true>(active_e))) {
                auto [ds, emitter_val] = scene->sample_emitter_direction(
                    si, sampler->next_2d(active_e), true, active_e);
                active_e &= neq(ds.pdf, 0.f);

                // Query the BSDF for that emitter-sampled direction
                Vector3f wo = si.to_local(ds.d);
                Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
                bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Determine density of sampling that same direction using BSDF sampling
                Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active_e);

                Float mis = select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));
                result[active_e] += mis * throughput * bsdf_val * emitter_val;
            }

            // ----------------------- BSDF sampling ----------------------

            // Sample BSDF * cos(theta)
            auto [bs, bsdf_val] = bsdf->sample(ctx, si, sampler->next_1d(active),
                                               sampler->next_2d(active), active);
            bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

            throughput = throughput * bsdf_val;
            active &= any(neq(depolarize(throughput), 0.f));
            if (none_or<false>(active))
                break;

            eta *= bs.eta;

            // Intersect the BSDF ray against the scene geometry
            ray = si.spawn_ray(si.to_world(bs.wo));
            SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray, active);

            /* Determine probability of having sampled that same
               direction using emitter sampling. */
            emitter = si_bsdf.emitter(scene, active);
            DirectionSample3f ds(si_bsdf, si);
            ds.object = emitter;

            if (any_or<true>(neq(emitter, nullptr))) {
                Float emitter_pdf =
                    select(neq(emitter, nullptr) && !has_flag(bs.sampled_type, BSDFFlags::Delta),
                           scene->pdf_emitter_direction(si, ds),
                           0.f);

                emission_weight = mis_weight(bs.pdf, emitter_pdf);
            }

            si = std::move(si_bsdf);
        }

        return { result, valid_ray };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("ReorderIntegrator[\n"
            "  max_depth = %i,\n"
            "  rr_depth = %i\n"
            "]", m_max_depth, m_rr_depth);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(ReorderIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(ReorderIntegrator, "PSS Reorder Path Tracer integrator");

NAMESPACE_END(mitsuba)
