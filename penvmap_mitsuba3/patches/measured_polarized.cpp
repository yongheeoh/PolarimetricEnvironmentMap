#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/tensor.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/microfacet.h>

/* Set the weight for cosine hemisphere sampling in relation to GGX sampling.
   Set to 1.0 in order to fully fall back to cosine sampling. */
#define COSINE_HEMISPHERE_PDF_WEIGHT 0.1f

NAMESPACE_BEGIN(mitsuba)

/**!

.. _bsdf-measured_polarized:

Measured polarized material (:monosp:`measured_polarized`)
----------------------------------------------------------

.. pluginparameters::

 * - filename
   - |string|
   - Filename of the material data file to be loaded

 * - alpha_sample
   - |float|
   - Specifies which roughness value should be used for the internal Microfacet
     importance sampling routine. (Default: 0.1)

 * - wavelength
   - |float|
   - Specifies if the material should only be rendered for just one specific wavelength.
     The valid range is between 450 and 650 nm.
     A value of -1 means the full spectrally-varying pBRDF will be used.
     (Default: -1, i.e. all wavelengths.)

This plugin allows rendering of polarized materials (pBRDFs) acquired as part of
"Image-Based Acquisition and Modeling of Polarimetric Reflectance" by Baek et al.
2020 (:cite:`Baek2020Image`). The required files for each material can
be found in the `corresponding database <http://vclab.kaist.ac.kr/siggraph2020/pbrdfdataset/kaistdataset.html>`_.

The dataset is made out of isotropic pBRDFs spanning a wide range of appearances:
diffuse/specular, metallic/dielectric, rough/smooth, and different color albedos,
captured in five wavelength ranges covering the visible spectrum from 450 to 650 nm.

Here are two example materials of gold, and a dielectric "fake" gold, together with
visualizations of the resulting Stokes vectors rendered with the
:ref:`Stokes <integrator-stokes>` integrator:

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/bsdf_measured_polarized_gold.jpg
   :caption: Measured gold material
.. subfigure:: ../../resources/data/docs/images/render/bsdf_measured_polarized_gold_stokes_s1.jpg
   :caption: ":math:`\mathbf{s}_1`": horizontal vs. vertical polarization
.. subfigure:: ../../resources/data/docs/images/render/bsdf_measured_polarized_gold_stokes_s2.jpg
   :caption: ":math:`\mathbf{s}_2`": positive vs. negative diagonal polarization
.. subfigure:: ../../resources/data/docs/images/render/bsdf_measured_polarized_gold_stokes_s3.jpg
   :caption: ":math:`\mathbf{s}_3`": right vs. left circular polarization
.. subfigend::
   :label: fig-stokes

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/bsdf_measured_polarized_fakegold.jpg
   :caption: Measured "fake" gold material
.. subfigure:: ../../resources/data/docs/images/render/bsdf_measured_polarized_fakegold_stokes_s1.jpg
   :caption: ":math:`\mathbf{s}_1`": horizontal vs. vertical polarization
.. subfigure:: ../../resources/data/docs/images/render/bsdf_measured_polarized_fakegold_stokes_s2.jpg
   :caption: ":math:`\mathbf{s}_2`": positive vs. negative diagonal polarization
.. subfigure:: ../../resources/data/docs/images/render/bsdf_measured_polarized_fakegold_stokes_s3.jpg
   :caption: ":math:`\mathbf{s}_3`": right vs. left circular polarization
.. subfigend::
   :label: fig-stokes

In the following example, the measured gold BSDF from the dataset is setup:

.. tabs::
    .. code-tab:: xml
        :name: measured-polarized

        <bsdf type="measured_polarized">
            <string name="filename" value="6_gold_inpainted.pbsdf"/>
            <float name="alpha_sample" value="0.02"/>
        </bsdf>

    .. code-tab:: python

        'type': 'measured_polarized',
        'filename': '6_gold_inpainted.pbsdf',
        'alpha_sample': 0.02

Internally, a sampling routine from the GGX Microfacet model is used in order to
importance sampling outgoing directions. The used GGX roughness value is exposed
here as a user parameter `alpha_sample` and should be set according to the
approximate roughness of the material to be rendered. Note that any value here
will result in a correct rendering but the level of noise can vary significantly.

.. warning::
    This BSDF is only supported in ``*_spectral_polarized`` variants.
*/
template <typename Float, typename Spectrum>
class MeasuredPolarized final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture, MicrofacetDistribution)

    using FloatStorage = DynamicBuffer<Float>;
    using Entries = dr::Array<Float, 16>;

    MeasuredPolarized(const Properties &props) : Base(props) {
        m_flags = BSDFFlags::GlossyReflection | BSDFFlags::FrontSide;
        m_components.push_back(m_flags);

        m_alpha_sample = props.get<ScalarFloat>("alpha_sample", 0.1f);
        m_wavelength = props.get<ScalarFloat>("wavelength", -1.f);

        if (!is_spectral_v<Spectrum> && m_wavelength == -1.f)
            Throw("In non-spectral modes, the measured polarized plugin can only render a specific wavelength specified by the `wavelength` parameter.");

        auto fs = file_resolver();
        fs::path file_path = fs->resolve(props.get<std::string_view>("filename"));
        m_name = file_path.filename().string();

        ref<TensorFile> tf = new TensorFile(file_path);

        auto theta_h = tf->field("theta_h");
        auto theta_d = tf->field("theta_d");
        auto phi_d   = tf->field("phi_d");
        auto wvls    = tf->field("wvls");
        auto pbrdf   = tf->field("M");

        if (!(theta_h.shape.size() == 2 &&
              theta_h.dtype == sj::Type::Float32 &&

              theta_d.shape.size() == 2 &&
              theta_d.dtype == sj::Type::Float32 &&

              phi_d.shape.size() == 2 &&
              phi_d.dtype == sj::Type::Float32 &&

              wvls.shape.size() == 1 &&
              wvls.dtype == sj::Type::UInt16 &&

              pbrdf.dtype == sj::Type::Float32 &&
              pbrdf.shape.size() == 6 &&
              pbrdf.shape[0] == phi_d.shape[1] &&
              pbrdf.shape[1] == theta_d.shape[1] &&
              pbrdf.shape[2] == theta_h.shape[1] &&
              pbrdf.shape[3] == wvls.shape[0]) &&
              pbrdf.shape[4] == 4 &&
              pbrdf.shape[5] == 4) {
            Throw("Invalid file structure: %s", tf->to_string());
        }

        m_size[0] = (uint32_t) phi_d.shape[1];
        m_size[1] = (uint32_t) theta_d.shape[1];
        m_size[2] = (uint32_t) theta_h.shape[1];
        m_size[3] = (uint32_t) wvls.shape[0];
        for (size_t k = 0; k < 4; ++k)
            if (m_size[k] < 2)
                Throw("Invalid file structure (an axis with fewer than 2 values): %s", tf->to_string());

        const ScalarFloat *angles[3] = { (const ScalarFloat *) phi_d.data,
                                         (const ScalarFloat *) theta_d.data,
                                         (const ScalarFloat *) theta_h.data };
        for (size_t k = 0; k < 3; ++k)
            m_axis[k] = dr::load<FloatStorage>(angles[k], m_size[k]);
        std::vector<ScalarFloat> wavelengths(m_size[3]);
        for (size_t i = 0; i < m_size[3]; ++i)
            wavelengths[i] = ScalarFloat(((uint16_t *) wvls.data)[i]);
        m_axis[3] = dr::load<FloatStorage>(wavelengths.data(), m_size[3]);

        m_data = dr::load<FloatStorage>((const ScalarFloat *) pbrdf.data,
                                        (size_t) m_size[0] * m_size[1] * m_size[2] * m_size[3] * 16);
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f &sample2,
                                             Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        active &= cos_theta_i > 0.f;

        BSDFSample3f bs = dr::zeros<BSDFSample3f>();
        if (unlikely(dr::none_or<false>(active) || !ctx.is_enabled(BSDFFlags::GlossyReflection)))
            return { bs, 0.f };

        MicrofacetDistribution distr(MicrofacetType::GGX,
                                     m_alpha_sample, m_alpha_sample, true);

        Float lobe_pdf_diffuse = COSINE_HEMISPHERE_PDF_WEIGHT;
        Mask sample_diffuse    = active && sample1 < lobe_pdf_diffuse,
             sample_microfacet = active && !sample_diffuse;

        Vector3f wo_diffuse    = warp::square_to_cosine_hemisphere(sample2);
        auto [m, unused] = distr.sample(si.wi, sample2);
        Vector3f wo_microfacet = reflect(si.wi, m);

        bs.wo[sample_diffuse]    = wo_diffuse;
        bs.wo[sample_microfacet] = wo_microfacet;

        bs.pdf = pdf(ctx, si, bs.wo, active);

        bs.sampled_component = 0;
        bs.sampled_type = +BSDFFlags::GlossyReflection;
        bs.eta = 1.f;

        Spectrum value = eval(ctx, si, bs.wo, active);
        return { bs, dr::select(active && bs.pdf > 0, value / bs.pdf, 0.f) };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);
        active &= (cos_theta_i > 0.f && cos_theta_o > 0.f);

        if (unlikely(dr::none_or<false>(active) || !ctx.is_enabled(BSDFFlags::GlossyReflection)))
            return 0.f;

        /* Due to the coordinate system rotations for polarization-aware
           pBSDFs below we need to know the propagation direction of light.
           In the following, light arrives along `-wo_hat` and leaves along
           `+wi_hat`. */
        Vector3f wo_hat = ctx.mode == TransportMode::Radiance ? wo : si.wi,
                 wi_hat = ctx.mode == TransportMode::Radiance ? si.wi : wo;

        /* We now transform both directions to the standard frame defined in
           Figure 3. Here, one of the directions is aligned with the x-axis. */
        Float phi_std = phi(wi_hat);
        Vector3f wo_std = rotate_vector(wo_hat, Vector3f(0,0,1), -phi_std),
                 wi_std = rotate_vector(wi_hat, Vector3f(0,0,1), -phi_std);

        /* This representation can be turned into the (isotropic) Rusinkiewicz
           parameterization. */
        auto [phi_d, theta_h, theta_d] = directions_to_rusinkiewicz(wo_std, wi_std);

        Spectrum value;
        if constexpr (is_polarized_v<Spectrum>) {
            /* The Stokes reference frame vector of this matrix lies in the plane
               of reflection. See Figure 4. */
            Vector3f zo_std = -wo_std,
                     to_std = dr::normalize(dr::cross(wo_std - wi_std, zo_std)),
                     yo_std = dr::normalize(dr::cross(to_std, zo_std)),
                     xo_std = dr::cross(yo_std, zo_std),
                     zi_std = wi_std,
                     ti_std = dr::normalize(dr::cross(wi_std - wo_std, zi_std)),
                     yi_std = dr::normalize(dr::cross(ti_std, zi_std)),
                     xi_std = dr::cross(yi_std, zi_std);

            if (m_wavelength == -1.f) {
                auto m = lookup(phi_d, theta_d, theta_h, si.wavelengths, active);
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        for (size_t k = 0; k < dr::size_v<UnpolarizedSpectrum>; ++k)
                            value(i, j)[k] = m[k][4 * i + j];
            } else {
                auto m = lookup(phi_d, theta_d, theta_h, dr::Array<Float, 1>(m_wavelength), active);
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        value(i, j) = m[0][4 * i + j];
            }

            /* Invalid configurations such as transmission directions are encoded as NaNs.
               Make sure these values don't end up in the interpolated value. */
            dr::masked(value, dr::any(dr::isnan(value(0,0)))) = 0.f;

            // Make sure intensity is non-negative
            value(0, 0) = dr::maximum(0.f, value(0,0));

            // Reverse phi rotation from above on Stokes reference frames
            Vector3f xo_hat = rotate_vector(xo_std, Vector3f(0, 0, 1), phi_std),
                     xi_hat = rotate_vector(xi_std, Vector3f(0, 0, 1), phi_std);

            /* Rotate in/out reference vector of `value` s.t. it aligns with the
               implicit Stokes bases of -wo_hat & wi_hat. */
            value = mueller::rotate_mueller_basis(value,
                                                  -wo_hat, xo_hat, mueller::stokes_basis(-wo_hat),
                                                   wi_hat, xi_hat, mueller::stokes_basis(wi_hat));
        } else {
            if (m_wavelength == -1.f) {
                auto m = lookup(phi_d, theta_d, theta_h, si.wavelengths, active);
                for (size_t k = 0; k < dr::size_v<UnpolarizedSpectrum>; ++k)
                    value[k] = m[k][0];
            } else {
                auto m = lookup(phi_d, theta_d, theta_h, dr::Array<Float, 1>(m_wavelength), active);
                value = Spectrum(m[0][0]);
            }

            // Make sure BRDF is non-negative
            value = dr::maximum(0.f, value);
        }

        return (value * cos_theta_o) & active;
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (unlikely(dr::none_or<false>(active) || !ctx.is_enabled(BSDFFlags::GlossyReflection)))
            return 0.f;

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        MicrofacetDistribution distr(MicrofacetType::GGX,
                                     m_alpha_sample, m_alpha_sample, true);

        Vector3f H = dr::normalize(wo + si.wi);

        Float pdf_diffuse = warp::square_to_cosine_hemisphere_pdf(wo);
        Float pdf_microfacet = distr.pdf(si.wi, H) / (4.f * dr::dot(wo, H));

        Float pdf = 0.f;
        pdf += pdf_diffuse * COSINE_HEMISPHERE_PDF_WEIGHT;
        pdf += pdf_microfacet * (1.f - COSINE_HEMISPHERE_PDF_WEIGHT);

        return dr::select(cos_theta_i > 0.f && cos_theta_o > 0.f, pdf, 0.f);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "MeasuredPolarized[" << std::endl
            << "  name = " << m_name << std::endl
            << "]";
        return oss.str();
    }

private:
    template <typename Vector3,
              typename Value = dr::value_t<Vector3>>
    Value phi(const Vector3 &v) const {
        Value p = dr::atan2(v.y(), v.x());
        dr::masked(p, p < 0) += 2.f*dr::Pi<Float>;
        return p;
    }

    template <typename Vector3, typename Value = dr::value_t<Vector3>>
    MI_INLINE
    Vector3 rotate_vector(const Vector3 &v, const Vector3 &axis_, Value angle) const {
        Vector3 axis = dr::normalize(axis_);
        auto [sin_angle, cos_angle] = dr::sincos(angle);
        return v*cos_angle + axis*dr::dot(v, axis)*(1.f - cos_angle) + sin_angle*dr::cross(axis, v);
    }

    template <typename Vector3, typename Value = dr::value_t<Vector3>>
    MI_INLINE
    std::tuple<Value, Value, Value> directions_to_rusinkiewicz(const Vector3 &i, const Vector3 &o) const {
        Vector3 h = dr::normalize(i + o);

        Vector3 n(0, 0, 1);
        Vector3 b = dr::normalize(dr::cross(n, h)),
                t = dr::normalize(dr::cross(b, h));

        Value td = dr::safe_acos(dr::dot(h, i)),
              th = dr::safe_acos(dr::dot(n, h));

        Vector3 i_prj = dr::normalize(i - dr::dot(i, h)*h);
        Value cos_phi_d = dr::clip(dr::dot(t, i_prj), -1.f, 1.f),
              sin_phi_d = dr::clip(dr::dot(b, i_prj), -1.f, 1.f);

        Value pd = dr::atan2(sin_phi_d, cos_phi_d);

        return std::make_tuple(pd, th, td);
    }

    std::pair<UInt32, Float> locate(size_t k, const Float &x, Mask active) const {
        UInt32 idx = math::find_interval<UInt32>(m_size[k], [&](UInt32 i) DRJIT_INLINE_LAMBDA {
            return dr::gather<Float>(m_axis[k], i, active) < x;
        });
        Float x0 = dr::gather<Float>(m_axis[k], idx, active),
              x1 = dr::gather<Float>(m_axis[k], idx + 1, active);
        return { idx, dr::clip((x - x0) / (x1 - x0), 0.f, 1.f) };
    }

    /// Mueller matrix entries (row-major) per wavelength, interpolated once
    /// for all 16 entries rather than per entry
    template <typename Wavelengths, size_t N = dr::size_v<Wavelengths>>
    dr::Array<Entries, N> lookup(const Float &phi_d, const Float &theta_d, const Float &theta_h,
                                 const Wavelengths &wavelengths, Mask active) const {
        const Float angle[3] = { phi_d, theta_d, theta_h };
        const uint32_t stride[3] = { m_size[1] * m_size[2] * m_size[3], m_size[2] * m_size[3], m_size[3] };
        Float w[3];
        UInt32 bin = 0;
        for (size_t k = 0; k < 3; ++k) {
            auto [idx, wk] = locate(k, angle[k], active);
            w[k] = wk;
            bin += idx * stride[k];
        }

        std::vector<Entries> band(m_size[3], dr::zeros<Entries>());
        for (uint32_t c = 0; c < 8; ++c) {
            Float weight = 1.f;
            UInt32 corner = bin;
            for (size_t k = 0; k < 3; ++k) {
                bool upper = (c >> k) & 1;
                weight *= upper ? w[k] : 1.f - w[k];
                if (upper)
                    corner += stride[k];
            }
            for (uint32_t b = 0; b < m_size[3]; ++b)
                band[b] = dr::fmadd(dr::gather<Entries>(m_data, corner + b, active), weight, band[b]);
        }

        dr::Array<Entries, N> result;
        for (size_t s = 0; s < N; ++s) {
            auto [idx, ws] = locate(3, wavelengths[s], active);
            Entries lo = band[0], hi = band[1];
            for (uint32_t b = 1; b + 1 < m_size[3]; ++b) {
                lo = dr::select(idx == b, band[b], lo);
                hi = dr::select(idx == b, band[b + 1], hi);
            }
            result[s] = dr::fmadd(lo, 1.f - ws, hi * ws);
        }
        return result;
    }

    MI_DECLARE_CLASS(MeasuredPolarized)
private:
    std::string m_name;
    ScalarFloat m_wavelength;
    ScalarFloat m_alpha_sample;
    uint32_t m_size[4];     // phi_d, theta_d, theta_h, band
    FloatStorage m_axis[4];
    FloatStorage m_data;    // [phi_d][theta_d][theta_h][band][4 x 4 row-major]

    MI_TRAVERSE_CB(Base, m_axis[0], m_axis[1], m_axis[2], m_axis[3], m_data)
};

MI_EXPORT_PLUGIN(MeasuredPolarized)
NAMESPACE_END(mitsuba)
