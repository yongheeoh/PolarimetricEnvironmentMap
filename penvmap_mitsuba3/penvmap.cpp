#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/bsphere.h>
#include <mitsuba/core/distr_2d.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/mueller.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/srgb.h>
#include <drjit/tensor.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/core/fstream.h>

NAMESPACE_BEGIN(mitsuba)


/**!

.. _emitter-envmap:

Polarized environment emitter (:monosp:`penvmap`)
--------------------------------------

.. pluginparameters::
 :extra-rows: 4

 * - filename
    - |string|
    - Filename of a single OpenEXR input image containing the :math:`\mathbf{s}_0` through :math:`\mathbf{s}_3` layers as RGB channels named :monosp:`S0.R`, :monosp:`S0.G`, :monosp:`S0.B`, ..., :monosp:`S3.B`.

 * - bitmap
     - :monosp:`Bitmap object`
     - Bitmap object containing a single multi-channel OpenEXR image with the :math:`\mathbf{s}_0` through :math:`\mathbf{s}_3` layers encoded as :monosp:`S0.R`, :monosp:`S0.G`, :monosp:`S0.B`, ..., :monosp:`S3.B`.

 * - filename_s0
   - |string|
    - Filename of an alternative :math:`\mathbf{s}_0` radiance-valued latitude-longitude input image.

 * - filename_s1
    - |string|
     - Filename of an alternative :math:`\mathbf{s}_1` polarized latitude-longitude input image.

 * - filename_s2 
    - |string|
     - Filename of an alternative :math:`\mathbf{s}_2` polarized latitude-longitude input image.

 * - filename_s3
    - |string|
     - Filename of an alternative :math:`\mathbf{s}_3` polarized latitude-longitude input image.

 * - bitmap_s0
     - :monosp:`Bitmap object`
     - Bitmap object containing an alternative :math:`\mathbf{s}_0` radiance-valued latitude-longitude image.

 * - bitmap_s1
     - :monosp:`Bitmap object`
     - Bitmap object containing an alternative :math:`\mathbf{s}_1` polarized latitude-longitude image.

 * - bitmap_s2
     - :monosp:`Bitmap object`
     - Bitmap object containing an alternative :math:`\mathbf{s}_2` polarized latitude-longitude image.

 * - bitmap_s3
     - :monosp:`Bitmap object`
     - Bitmap object containing an alternative :math:`\mathbf{s}_3` polarized latitude-longitude image.

 * - scale
   - |Float|
   - A scale factor that is applied to the radiance values stored in the input image. (Default: 1.0)
   - |exposed|, |differentiable|

 * - to_world
   - |transform|
   - Specifies an optional emitter-to-world transformation.  (Default: none, i.e. emitter space = world space)
   - |exposed|

 * - mis_compensation
   - |bool|
   - Compensate sampling for the presence of other Monte Carlo techniques that
     will be combined using multiple importance sampling (MIS)? This is
     extremely cheap to do and can slightly reduce variance. (Default: false)

 * - data
     - |tensor|
     - Packed tensor array containing the polarized radiance-valued data with shape :math:`[h, w, 4, 3]`, where axis 2 stores :math:`\mathbf{s}_0` through :math:`\mathbf{s}_3`.
     - |exposed|, |differentiable|, |discontinuous|
      
This plugin provides a polarized HDRI (high dynamic range imaging) environment map,
which is a type of light source that is well-suited for representing "natural"
illumination.

An alternative four-input configuration accepts either filenames via :monosp:`filename_s0`
through :monosp:`filename_s3`, or already-loaded Mitsuba bitmaps via
:monosp:`bitmap_s0` through :monosp:`bitmap_s3`. These per-Stokes keys must not be mixed
with the single-file :monosp:`filename` or single-bitmap :monosp:`bitmap`
configuration.

The implementation loads a captured illumination environment from a image in
latitude-longitude format and turns it into an infinitely distant emitter. The
conventions of this mapping are shown in this image:

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/emitter/emitter_envmap_example.jpg
   :caption: The museum environment map by Bernhard Vogl that is used in
             many example renderings in this documentation.
.. subfigure:: ../../resources/data/docs/images/emitter/emitter_envmap_axes.jpg
   :caption: Coordinate conventions used when mapping the input image onto the sphere.
.. subfigend::
   :label: fig-penvmap-mapping

.. tabs::
    .. code-tab:: xml
        :name: penvmap-light

        <emitter type="penvmap">
            <string name="filename" value="textures/museum_stokes.exr"/>
        </emitter>

    .. code-tab:: python

        stokes_bitmap = mi.Bitmap('textures/museum_stokes.exr')

        'type': 'penvmap',
        'bitmap': stokes_bitmap,

 */

template <typename Float, typename Spectrum>
class PolarizedEnvironmentMapEmitter final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_to_world)
    MI_IMPORT_TYPES(Scene, Shape, Texture)

    using Warp = Hierarchical2D<Float, 0>;

    /* In RGB variants: 3-channel array for R, G, and B components
       In spectral variants: 4-channel array for polynomial coefficients & scale */
    static constexpr uint32_t StokesCount = 4;
    using PixelData = dr::Array<Float, is_spectral_v<Spectrum> ? 4 : 3>;
    using ScalarPixelData = dr::Array<ScalarFloat, is_spectral_v<Spectrum> ? 4 : 3>;

    PolarizedEnvironmentMapEmitter(const Properties &props) : Base(props) {
        if (is_spectral_v<Spectrum>) 
            Throw("spectral variants are not supported yet.");

        /* Until `set_scene` is called, we have no information
           about the scene and default to the unit bounding sphere. */
        m_bsphere = BoundingSphere3f(ScalarPoint3f(0.f), 1.f);

        std::vector<ref<Bitmap>> vec_bitmap;
        vec_bitmap.reserve(4);
        m_vec_filename.reserve(4);

        bool has_filename = props.has_property("filename");
        bool has_bitmap = props.has_property("bitmap");

        if (has_filename || has_bitmap) {
            if (has_filename && has_bitmap)
                Throw("Cannot specify both \"bitmap\" and \"filename\".");

            for (int i = 0; i < 4; ++i) {
                std::string suffix = "_s" + std::to_string(i);
                if (props.has_property("filename" + suffix) || props.has_property("bitmap" + suffix))
                    Throw("Cannot specify a single multi-channel input together with the alternative per-Stokes polarized environment map inputs.");
            }

            ref<Bitmap> bitmap;
            std::string base_filename;

            if (has_bitmap) {
                ref<Object> other = props.get<ref<Object>>("bitmap");
                Bitmap *b = dynamic_cast<Bitmap *>(other.get());
                if (!b)
                    Throw("Property \"bitmap\" must be a Bitmap instance.");
                bitmap = b;
                base_filename = "<Bitmap>";
            } else {
                FileResolver *fs = file_resolver();
                fs::path file_path = fs->resolve(props.get<std::string_view>("filename"));
                base_filename = file_path.filename().string();
                bitmap = new Bitmap(file_path);
            }

            auto split_bitmaps = bitmap->split();

            std::vector<ref<Bitmap>> layer_bitmaps(4);
            std::vector<std::string> layer_filenames(4);
            std::vector<bool> layer_found(4, false);

            for (const auto &layer : split_bitmaps) {
                const std::string &prefix = layer.first;
                if (prefix == "<root>")
                    continue;

                int layer_index = -1;
                for (int i = 0; i < 4; ++i) {
                    if (prefix == "S" + std::to_string(i)) {
                        layer_index = i;
                        break;
                    }
                }

                if (layer_index == -1)
                    Throw("\"%s\": unexpected layer \"%s\". Expected S0, S1, S2, or S3.",
                          base_filename, prefix);

                if (layer_found[layer_index])
                    Throw("\"%s\": duplicate layer \"%s\".", base_filename, prefix);

                Bitmap::PixelFormat pixel_format = layer.second->pixel_format();
                if (pixel_format != Bitmap::PixelFormat::RGB && pixel_format != Bitmap::PixelFormat::RGBA)
                    Throw("\"%s\": layer \"%s\" must contain RGB data.", base_filename, prefix);

                layer_bitmaps[layer_index] = layer.second;
                layer_filenames[layer_index] = base_filename + ":" + prefix;
                layer_found[layer_index] = true;
            }

            for (int i = 0; i < 4; ++i) {
                if (!layer_found[i])
                    Throw("\"%s\": missing required layer \"S%i\".", base_filename, i);
                vec_bitmap.push_back(layer_bitmaps[i]);
                m_vec_filename.push_back(layer_filenames[i]);
            }
        } else {
            for (int i = 0; i < 4; ++i) {
                std::string _si = "_s" + std::to_string(i);
                // Creates a Bitmap texture directly from an existing Bitmap object
                if (props.has_property("bitmap" + _si)) {
                    if (props.has_property("filename" + _si))
                        Throw("Cannot specify both \"bitmap%s\" and \"filename%s\".", _si, _si);
                    // Note: ref-counted, so we don't have to worry about lifetime
                    ref<Object> other = props.get<ref<Object>>("bitmap" + _si);
                    Bitmap *b = dynamic_cast<Bitmap *>(other.get());
                    if (!b)
                        Throw("Property \"bitmap%s\" must be a Bitmap instance.", _si);
                    m_vec_filename.push_back("");
                    vec_bitmap.push_back(b);
                } else if (props.has_property("filename" + _si)) {
                    FileResolver *fs = file_resolver();
                    fs::path file_path = fs->resolve(props.get<std::string_view>("filename" + _si));
                    m_vec_filename.push_back(file_path.filename().string());
                    vec_bitmap.push_back(new Bitmap(file_path));
                } else {
                    Throw("Property \"bitmap%s\" or \"filename%s\" must be specified.", _si, _si);
                }
            }
        }

        const uint32_t width_0 = vec_bitmap[0]->width();
        const uint32_t height_0 = vec_bitmap[0]->height();
        for (int i = 0; i < 4; ++i) {
            const uint32_t width_i = vec_bitmap[i]->width();
            const uint32_t height_i = vec_bitmap[i]->height();
            if (width_i < 2 || height_i < 3)
                Throw("\"%s\": the environment map resolution must be at least "
                      "2x3 pixels", m_vec_filename[i].empty() ? "<Bitmap>" : m_vec_filename[i]);

            if (width_i != width_0 || height_i != height_0)
                Throw("All polarized environment map images must have the same resolution, "
                      "but found %ux%u for \"%s\" and %ux%u for \"%s\"",
                      width_0, height_0, m_vec_filename[0].empty() ? "<Bitmap>" : m_vec_filename[0],
                      width_i, height_i, m_vec_filename[i].empty() ? "<Bitmap>" : m_vec_filename[i]);
        }

        const size_t pixel_width = is_spectral_v<Spectrum> ? 4 : 3;
        ScalarVector2u res(width_0 + 1, height_0);
        std::vector<ScalarFloat> packed_data((size_t) res.x() * (size_t) res.y() * StokesCount * pixel_width);

        for (int i = 0; i < (int) StokesCount; ++i) {
            ref<Bitmap> bitmap = vec_bitmap[i];

            /* Convert to linear RGBA float bitmap, will undergo further
            conversion into coefficients of a spectral upsampling model below */
            Bitmap::PixelFormat pixel_format = Bitmap::PixelFormat::RGB;
            if constexpr (is_spectral_v<Spectrum>)
                pixel_format = Bitmap::PixelFormat::RGBA;
            bitmap = bitmap->convert(pixel_format, struct_type_v<Float>, false);

            ScalarFloat *in_ptr = (ScalarFloat *) bitmap->data();
            ScalarFloat theta_scale = 1.f / (bitmap->size().y() - 1) * dr::Pi<Float>;

            ScalarFloat luminance_offset = 0.f;

            // s0: Luminance image used for importance sampling
            if (i == 0) {
                std::unique_ptr<ScalarFloat[]> luminance_data(new ScalarFloat[dr::prod(res)]);
                ScalarFloat *lum_ptr = (ScalarFloat *) luminance_data.get();

                /* "MIS Compensation: Optimizing Sampling Techniques in Multiple
                    Importance Sampling" Ondrej Karlik, Martin Sik, Petr Vivoda, Tomas
                    Skrivan, and Jaroslav Krivanek. SIGGRAPH Asia 2019 */
                if (props.get<bool>("mis_compensation", false)) {
                    ScalarFloat min_lum = 0.f;
                    double lum_accum_d = 0.0;

                    for (size_t y = 0; y < bitmap->size().y(); ++y) {
                        for (size_t x = 0; x < bitmap->size().x(); ++x) {
                            ScalarColor3f rgb = dr::load<ScalarVector3f>(in_ptr);
                            ScalarFloat lum = luminance(rgb);
                            min_lum = dr::minimum(min_lum, lum);
                            lum_accum_d += (double) lum;
                            in_ptr += 4;
                        }
                    }
                    in_ptr = (ScalarFloat *) bitmap->data();

                    luminance_offset = ScalarFloat(lum_accum_d / dr::prod(bitmap->size()));

                    /* Be wary of constant environment maps: average and minimum
                        should be sufficiently different */
                    if (luminance_offset - min_lum <= 0.01f * luminance_offset)
                        luminance_offset = 0.f; // disable
                }

                for (size_t y = 0; y < bitmap->size().y(); ++y) {
                    ScalarFloat sin_theta = dr::sin(y * theta_scale);
                    for (size_t x = 0; x < bitmap->size().x(); ++x) {
                        ScalarColor3f rgb = dr::load<ScalarVector3f>(in_ptr);
                        ScalarFloat lum = luminance(rgb);
                        lum = dr::maximum(lum - luminance_offset, 0.f);
                        *lum_ptr++ = lum * sin_theta;
                        in_ptr += pixel_width;
                    }
                    // Last column of pixels mirrors first
                    ScalarFloat temp = *(lum_ptr - bitmap->size().x());
                    *lum_ptr++ = temp;
                }
                m_warp = Warp(luminance_data.get(), res);

                // Back to the original bitmap pointer
                in_ptr = (ScalarFloat *) bitmap->data();
            }

            for (size_t y = 0; y < bitmap->size().y(); ++y) {
                for (size_t x = 0; x < bitmap->size().x(); ++x) {
                    ScalarColor3f rgb = dr::load<ScalarVector3f>(in_ptr);
                    ScalarFloat lum = luminance(rgb);

                    ScalarPixelData coeff;
                    if constexpr (is_monochromatic_v<Spectrum>) {
                        coeff = ScalarPixelData(lum);
                    } else if constexpr (is_rgb_v<Spectrum>) {
                        coeff = rgb;
                    } else {
                        static_assert(is_spectral_v<Spectrum>);
                        /* Evaluate the spectral upsampling model. This requires a
                            reflectance value (colors in [0, 1]) which is accomplished here by
                            scaling. We use a color where the highest component is 50%,
                            which generally yields a fairly smooth spectrum. */
                        ScalarFloat scale = dr::max(dr::abs(rgb)) * 2.f;
                        ScalarColor3f rgb_norm = rgb / dr::maximum(1e-8f, scale);
                        coeff = dr::concat((ScalarColor3f) srgb_model_fetch(dr::abs(rgb_norm)) * dr::sign(rgb_norm),
                                           dr::Array<ScalarFloat, 1>(scale));
                    }

                    ScalarFloat *dst = packed_data.data() + packed_scalar_offset(y * (size_t) res.x() + x,
                                                                                 (size_t) i,
                                                                                 pixel_width);
                    dr::store(dst, coeff);
                    in_ptr += pixel_width;
                }

                // Last column of pixels mirrors first
                const size_t row_start = y * (size_t) res.x();
                ScalarFloat *src = packed_data.data() + packed_scalar_offset(row_start,
                                                                              (size_t) i,
                                                                              pixel_width);
                ScalarFloat *dst = packed_data.data() + packed_scalar_offset(row_start + bitmap->size().x(),
                                                                              (size_t) i,
                                                                              pixel_width);
                dr::store(dst, dr::load<ScalarPixelData>(src));
            }
        }

        size_t shape[4] = { (size_t) res.y(), (size_t) res.x(), StokesCount, pixel_width };
        m_data = TensorXf(packed_data.data(), 4, shape);

        m_scale = props.get<ScalarFloat>("scale", 1.f);
        m_d65 = Texture::D65(1.f);
        m_flags = EmitterFlags::Infinite | EmitterFlags::SpatiallyVarying;

        parameters_changed({"data"});
    }

    void traverse(TraversalCallback *cb) override {
        Base::traverse(cb);
        cb->put("scale",     m_scale,         ParamFlags::Differentiable);
        cb->put("data",      m_data,          ParamFlags::Differentiable | ParamFlags::Discontinuous);
        cb->put("to_world",  m_to_world,      ParamFlags::NonDifferentiable);
    }

    void parameters_changed(const std::vector<std::string> &keys = {}) override {
        if (keys.empty() || string::contains(keys, "data")) {
            if (m_data.ndim() != 4)
                Throw("Environment map data has dimension %lu, expected 4", m_data.ndim());

            if (m_data.shape(2) != StokesCount)
                Throw("Environment map data has %lu Stokes channels, expected %u", m_data.shape(2), StokesCount);

            if constexpr (is_spectral_v<Spectrum>) {
                if (m_data.shape(3) != 4)
                    Throw("Environment map data has %lu channels, expected 4", m_data.shape(3));
            } else {
                if (m_data.shape(3) != 3)
                    Throw("Environment map data has %lu channels, expected 3", m_data.shape(3));
            }

            ScalarVector2u res = resolution();
            size_t pixel_width = is_spectral_v<Spectrum> ? 4 : 3;

            if constexpr (dr::is_jit_v<Float>) {
                // Enforce horizontal continuity per row and per Stokes component.
                UInt32 row_stokes = dr::arange<UInt32>((uint32_t) (res.y() * StokesCount));
                UInt32 row_index = (row_stokes / UInt32(StokesCount)) * UInt32(res.x() * StokesCount)
                                 + (row_stokes % UInt32(StokesCount));
                UInt32 boundary_offset = UInt32((res.x() - 1) * StokesCount);
                PixelData v0 = dr::gather<PixelData>(m_data.array(), row_index);
                PixelData v1 = dr::gather<PixelData>(m_data.array(), row_index + boundary_offset);
                PixelData v01 = .5f * (v0 + v1);
                dr::scatter(m_data.array(), v01, row_index);
                dr::scatter(m_data.array(), v01, row_index + boundary_offset);
            }

            auto &&data = dr::migrate(m_data.array(), JitBackend::None);

            if constexpr (dr::is_jit_v<Float>)
                dr::sync_thread();

            std::unique_ptr<ScalarFloat[]> luminance_data(
                new ScalarFloat[dr::prod(res)]);

            ScalarFloat *base_ptr = (ScalarFloat *) data.data(),
                        *lum_ptr  = (ScalarFloat *) luminance_data.get();

            constexpr bool is_aligned = ScalarPixelData::Size == 4;
            const size_t row_stride = (size_t) res.x() * StokesCount * pixel_width;
            const size_t col_stride = StokesCount * pixel_width;
            const size_t stokes_stride = pixel_width;

            if constexpr (!dr::is_jit_v<Float>) {
                for (size_t y = 0; y < res.y(); ++y) {
                    ScalarFloat *row_ptr = base_ptr + y * row_stride;
                    for (size_t stokes_i = 0; stokes_i < StokesCount; ++stokes_i) {
                        ScalarFloat *ptr0 = row_ptr + stokes_i * stokes_stride;
                        ScalarFloat *ptr1 = ptr0 + (res.x() - 1) * col_stride;
                        ScalarPixelData v0, v1;
                        if constexpr (is_aligned) {
                            v0 = dr::load_aligned<ScalarPixelData>(ptr0);
                            v1 = dr::load_aligned<ScalarPixelData>(ptr1);
                        } else {
                            v0 = dr::load<ScalarPixelData>(ptr0);
                            v1 = dr::load<ScalarPixelData>(ptr1);
                        }
                        ScalarPixelData v01 = .5f * (v0 + v1);

                        if constexpr (is_aligned) {
                            dr::store_aligned(ptr0, v01);
                            dr::store_aligned(ptr1, v01);
                        } else {
                            dr::store(ptr0, v01);
                            dr::store(ptr1, v01);
                        }
                    }
                }
            }

            ScalarFloat theta_scale = 1.f / (res.y() - 1) * dr::Pi<Float>;
            for (size_t y = 0; y < res.y(); ++y) {
                ScalarFloat sin_theta = dr::sin(y * theta_scale);
                for (size_t x = 0; x < res.x(); ++x) {
                    ScalarFloat *ptr = base_ptr + packed_scalar_offset(y * (size_t) res.x() + x,
                                                                       0,
                                                                       pixel_width);
                    ScalarPixelData coeff;
                    if constexpr (is_aligned)
                        coeff = dr::load_aligned<ScalarPixelData>(ptr);
                    else
                        coeff = dr::load<ScalarPixelData>(ptr);

                    ScalarFloat lum;
                    if constexpr (is_monochromatic_v<Spectrum>) {
                        lum = coeff.x();
                    } else if constexpr (is_rgb_v<Spectrum>) {
                        lum = luminance(ScalarColor3f(coeff));
                    } else {
                        static_assert(is_spectral_v<Spectrum>);
                        lum = srgb_model_mean(dr::head<3>(coeff)) * coeff.w();
                    }

                    *lum_ptr++ = lum * sin_theta;
                }
            }

            m_warp = Warp(luminance_data.get(), res);
        }

        Base::parameters_changed(keys);
    }

    void set_scene(const Scene *scene) override {
        if (scene->bbox().valid()) {
            ScalarBoundingSphere3f scene_sphere =
                scene->bbox().bounding_sphere();
            m_bsphere = BoundingSphere3f(scene_sphere.center, scene_sphere.radius);
            m_bsphere.radius =
                dr::maximum(math::RayEpsilon<Float>,
                        m_bsphere.radius * (1.f + math::RayEpsilon<Float>));
        } else {
            m_bsphere.center = 0.f;
            m_bsphere.radius = math::RayEpsilon<Float>;
        }

        dr::make_opaque(m_bsphere.center, m_bsphere.radius);
    }

    Spectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        Vector3f v = m_to_world.value().inverse() * (-si.wi);

        // Convert to latitude-longitude texture coordinates
        Point2f uv = Point2f(dr::atan2(v.x(), -v.z()) * dr::InvTwoPi<Float>,
                             dr::safe_acos(v.y()) * dr::InvPi<Float>);

        return to_world_basis(eval_spectrum(uv, si.wavelengths, active), v, -si.wi);
    }

    std::pair<Ray3f, Spectrum> sample_ray(Float time, Float wavelength_sample,
                                          const Point2f &sample2,
                                          const Point2f &sample3,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);

        // 1. Sample spatial component
        Point2f offset = warp::square_to_uniform_disk_concentric(sample2);

        // 2. Sample directional component
        auto [uv, pdf] = m_warp.sample(sample3, nullptr, active);
        uv.x() += .5f / (m_data.shape(1) - 1);

        active &= pdf > 0.f;

        Float theta = uv.y() * dr::Pi<Float>,
              phi   = uv.x() * dr::TwoPi<Float>;

        Vector3f d = dr::sphdir(theta, phi);
        d = Vector3f(d.y(), d.z(), -d.x());

        Float inv_sin_theta = dr::safe_rsqrt(dr::square(d.x()) + dr::square(d.z()));
        pdf *= inv_sin_theta * dr::InvTwoPi<Float> * dr::InvPi<Float>;

        // Unlike \ref sample_direction, ray goes from the envmap toward the scene
        Vector3f d_global = m_to_world.value() * -d;

        // Compute ray origin
        Vector3f perpendicular_offset =
            Frame3f(d).to_world(Vector3f(offset.x(), offset.y(), 0));
        Point3f origin =
            m_bsphere.center + (perpendicular_offset - d_global) * m_bsphere.radius;

        // 3. Sample spectral component (weight accounts for radiance)
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.t    = 0.f;
        si.time = time;
        si.p    = origin;
        si.uv   = uv;
        auto [wavelengths, weight] =
            sample_wavelengths(si, wavelength_sample, active);

        Float r2 = dr::square(m_bsphere.radius);
        Ray3f ray(origin, d_global, time, wavelengths);
        weight = to_world_basis(weight, d, -d_global);
        weight *= dr::Pi<Float> * r2 / pdf;

        return std::make_pair(ray, weight & active);
    }

    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f &it, const Point2f &sample,
                     Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);

        auto [uv, pdf] = m_warp.sample(sample, nullptr, active);
        uv.x() += .5f / (m_data.shape(1) - 1);
        active &= pdf > 0.f;

        Float theta = uv.y() * dr::Pi<Float>,
              phi   = uv.x() * dr::TwoPi<Float>;

        Vector3f d = dr::sphdir(theta, phi);
        d = Vector3f(d.y(), d.z(), -d.x());

        // Needed when the reference point is on the sensor, which is not part of the bbox
        Float radius = dr::maximum(m_bsphere.radius, dr::norm(it.p - m_bsphere.center));
        Float dist = 2.f * radius;

        Float inv_sin_theta = dr::safe_rsqrt(dr::maximum(
            dr::square(d.x()) + dr::square(d.z()), dr::square(dr::Epsilon<Float>)));

        d = m_to_world.value() * d;

        DirectionSample3f ds;
        ds.p       = it.p + d * dist;
        ds.n       = -d;
        ds.uv      = uv;
        ds.time    = it.time;
        ds.pdf     = dr::select(
            active,
            pdf * inv_sin_theta * (1.f / (2.f * dr::square(dr::Pi<Float>))),
            0.f
        );
        ds.delta   = false;
        ds.emitter = this;
        ds.d       = d;
        ds.dist    = dist;

        auto weight =
            depolarizer<Spectrum>(eval_spectrum(uv, it.wavelengths, active)) /
            ds.pdf;

        return { ds, weight & active };
    }

    Float pdf_direction(const Interaction3f & /*it*/, const DirectionSample3f &ds,
                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);

        Vector3f d = m_to_world.value().inverse() * ds.d;

        // Convert to latitude-longitude texture coordinates
        Point2f uv = Point2f(dr::atan2(d.x(), -d.z()) * dr::InvTwoPi<Float>,
                             dr::safe_acos(d.y()) * dr::InvPi<Float>);
        uv.x() -= .5f / (m_data.shape(1) - 1u);
        uv -= dr::floor(uv);

        Float inv_sin_theta = dr::safe_rsqrt(dr::maximum(
            dr::square(d.x()) + dr::square(d.z()), dr::square(dr::Epsilon<Float>)));

        return m_warp.eval(uv) * inv_sin_theta * (1.f / (2.f * dr::square(dr::Pi<Float>)));
    }

    Spectrum eval_direction(const Interaction3f &it,
                            const DirectionSample3f &ds,
                            Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        Vector3f v = m_to_world.value().inverse() * ds.d;
        return to_world_basis(eval_spectrum(ds.uv, it.wavelengths, active), v, ds.d);
    }

    std::pair<Wavelength, Spectrum>
    sample_wavelengths(const SurfaceInteraction3f &si, Float sample,
                       Mask active) const override {
        auto [wavelengths, weight] = m_d65->sample_spectrum(
            si, math::sample_shifted<Wavelength>(sample), active);

        return { wavelengths,
                 weight * eval_spectrum(si.uv, wavelengths, active, false) };
    }

    std::pair<PositionSample3f, Float>
    sample_position(Float /*time*/, const Point2f & /*sample*/,
                    Mask /*active*/) const override {
        if constexpr (dr::is_jit_v<Float>) {
            /* Do not throw an exception in JIT-compiled variants. This
               function might be invoked by DrJit's virtual function call
               recording mechanism despite not influencing any actual
               calculation. */
            return { dr::zeros<PositionSample3f>(), dr::NaN<Float> };
        } else {
            NotImplementedError("sample_position");
        }
    }

    ScalarBoundingBox3f bbox() const override {
        /* This emitter does not occupy any particular region
           of space, return an invalid bounding box */
        return ScalarBoundingBox3f();
    }

    std::string to_string() const override {
        ScalarVector2u res = resolution();
        std::ostringstream oss;
        oss << "PolarizedEnvironmentMapEmitter[" << std::endl;
        for (int i = 0; i < 4; ++i) {
            if (!m_vec_filename[i].empty())
                oss << "  filename_s" << i << " = \"" << m_vec_filename[i] << "\"," << std::endl;
        }
        oss << "  res = \"" << res << "\"," << std::endl
            << "  bsphere = " << string::indent(m_bsphere) << std::endl
            << "]";
        return oss.str();
    }

protected:
    static UInt32 packed_stokes_index(const UInt32 &spatial_index, uint32_t stokes_i) {
        return dr::fmadd(spatial_index, UInt32(StokesCount), UInt32(stokes_i));
    }

    static size_t packed_scalar_offset(size_t spatial_index, size_t stokes_i, size_t pixel_width) {
        return (spatial_index * StokesCount + stokes_i) * pixel_width;
    }

    ScalarVector2u resolution() const {
        return { m_data.shape(1), m_data.shape(0) };
    }

    UnpolarizedSpectrum eval_unpolarizedspectrum(Point2f uv, const Wavelength &wavelengths,
                                                 Mask active, bool include_whitepoint = true, int stokes_i = 0) const {
        ScalarVector2u res = resolution();

        uv.x() -= .5f / (res.x() - 1u);
        uv -= dr::floor(uv);
        uv *= Vector2f(res - 1u);

        Point2u pos = dr::minimum(Point2u(uv), res - 2u);

        Point2f w1 = uv - Point2f(pos),
                w0 = 1.f - w1;

        const uint32_t width = res.x();
        UInt32 index = dr::fmadd(pos.y(), UInt32(width), UInt32(pos.x()));
        UInt32 packed_index = packed_stokes_index(index, (uint32_t) stokes_i);
        UInt32 col_stride = UInt32(StokesCount);
        UInt32 row_stride = UInt32(width * StokesCount);

        PixelData v00 = dr::gather<PixelData>(m_data.array(), packed_index, active),
              v10 = dr::gather<PixelData>(m_data.array(), packed_index + col_stride, active),
              v01 = dr::gather<PixelData>(m_data.array(), packed_index + row_stride, active),
              v11 = dr::gather<PixelData>(m_data.array(), packed_index + row_stride + col_stride, active);

        if constexpr (is_spectral_v<Spectrum>) {
            UnpolarizedSpectrum s00, s10, s01, s11, s0, s1, s;
            Float f0, f1, f;

            s00 = srgb_model_eval<UnpolarizedSpectrum>(dr::head<3>(v00), wavelengths);
            s10 = srgb_model_eval<UnpolarizedSpectrum>(dr::head<3>(v10), wavelengths);
            s01 = srgb_model_eval<UnpolarizedSpectrum>(dr::head<3>(v01), wavelengths);
            s11 = srgb_model_eval<UnpolarizedSpectrum>(dr::head<3>(v11), wavelengths);

            s0  = dr::fmadd(w0.x(), s00, w1.x() * s10);
            s1  = dr::fmadd(w0.x(), s01, w1.x() * s11);
            f0  = dr::fmadd(w0.x(), v00.w(), w1.x() * v10.w());
            f1  = dr::fmadd(w0.x(), v01.w(), w1.x() * v11.w());

            s   = dr::fmadd(w0.y(), s0, w1.y() * s1);
            f   = dr::fmadd(w0.y(), f0, w1.y() * f1);

            UnpolarizedSpectrum result = s * f * m_scale;

            if (include_whitepoint) {
                SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
                si.wavelengths = wavelengths;
                result *= m_d65->eval(si, active);
            }

            return result;
        } else {
            DRJIT_MARK_USED(wavelengths);
            PixelData v0 = dr::fmadd(w0.x(), v00, w1.x() * v10),
                      v1 = dr::fmadd(w0.x(), v01, w1.x() * v11),
                      v  = dr::fmadd(w0.y(), v0, w1.y() * v1);

            if constexpr (is_monochromatic_v<Spectrum>)
                return dr::head<1>(v) * m_scale;
            else
                return v * m_scale;
        }
    }

    Spectrum eval_spectrum(Point2f uv, const Wavelength &wavelengths,
                           Mask active, bool include_whitepoint = true) const {
        if constexpr (is_polarized_v<Spectrum>) {
            Spectrum result(0.f);         
            for (int i = 0; i < 4; ++i) {
                UnpolarizedSpectrum value = eval_unpolarizedspectrum(uv, wavelengths, active, include_whitepoint, i);
                result(i, 0) = value;
            }
            return result;
        } else {
            return eval_unpolarizedspectrum(uv, wavelengths, active, include_whitepoint);
        }
    }

    /* The Stokes vector stored for the emitter-space direction v is expressed
       in mueller::stokes_basis(-v). That basis turns with to_world, so re-express
       the vector in Mitsuba's basis mueller::stokes_basis(-d) of the world-space
       direction d = to_world * v. */
    Spectrum to_world_basis(const Spectrum &value, const Vector3f &v, const Vector3f &d) const {
        if constexpr (is_polarized_v<Spectrum>) {
            Vector3f basis = m_to_world.value() * mueller::stokes_basis(-v);
            return mueller::rotate_stokes_basis(-d, basis, mueller::stokes_basis(-d)) * value;
        } else {
            DRJIT_MARK_USED(v);
            DRJIT_MARK_USED(d);
            return value;
        }
    }

    MI_DECLARE_CLASS(PolarizedEnvironmentMapEmitter)
protected:
    std::vector<std::string> m_vec_filename;
    BoundingSphere3f m_bsphere;
    TensorXf m_data;
    Warp m_warp;
    ref<Texture> m_d65;
    Float m_scale;

    MI_TRAVERSE_CB(Base, m_bsphere, m_data, m_warp, m_d65, m_scale)
};

MI_EXPORT_PLUGIN(PolarizedEnvironmentMapEmitter)
NAMESPACE_END(mitsuba)
