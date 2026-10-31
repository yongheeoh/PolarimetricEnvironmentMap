package kr.penvmap.capture

import android.animation.ValueAnimator
import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Canvas
import android.graphics.Outline
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.drawable.Drawable
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.RippleDrawable
import android.os.Build
import android.text.InputType
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.ViewOutlineProvider
import android.view.WindowInsets
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.Switch
import android.widget.TextView

/** Colours of the app: a dark, low-glare theme for working outdoors next to the camera. */
object Palette {
    const val BG = 0xFF0A0D12.toInt()
    const val SURFACE = 0xFF131820.toInt()
    const val SURFACE2 = 0xFF1B2230.toInt()
    const val STROKE = 0xFF283245.toInt()
    const val TEXT = 0xFFE8EDF4.toInt()
    const val TEXT2 = 0xFF9AA6B8.toInt()
    const val TEXT3 = 0xFF677387.toInt()
    const val BLUE = 0xFF3D8BFF.toInt()
    const val TEAL = 0xFF1FC9A5.toInt()
    const val AMBER = 0xFFFFB547.toInt()
    const val RED = 0xFFFF5C5C.toInt()
    const val GREEN = 0xFF35D49A.toInt()

    fun level(l: CaptureController.Level) = when (l) {
        CaptureController.Level.OK -> GREEN
        CaptureController.Level.WARN -> AMBER
        CaptureController.Level.ERROR -> RED
        CaptureController.Level.INFO -> TEXT2
    }
}

/**
 * Small view toolkit for the code-built screens (no XML, no androidx): rounded cards, pill buttons with
 * ripples, status chips, segmented controls, a thin progress line.
 */
class Kit(val ctx: Context) {
    private val density = ctx.resources.displayMetrics.density

    fun dp(v: Float): Int = (v * density + 0.5f).toInt()
    fun dp(v: Int): Int = dp(v.toFloat())

    val medium: Typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
    val mono: Typeface = Typeface.create("monospace", Typeface.NORMAL)

    fun rounded(fill: Int, radiusDp: Float, stroke: Int? = null, strokeDp: Float = 1f) = GradientDrawable().apply {
        setColor(fill)
        cornerRadius = dp(radiusDp).toFloat()
        if (stroke != null) setStroke(dp(strokeDp), stroke)
    }

    fun ripple(content: Drawable, radiusDp: Float, color: Int = 0x33FFFFFF): Drawable =
        RippleDrawable(ColorStateList.valueOf(color), content, rounded(0xFFFFFFFF.toInt(), radiusDp))

    fun text(s: CharSequence = "", size: Float = 14f, color: Int = Palette.TEXT, bold: Boolean = false, mono: Boolean = false) =
        TextView(ctx).apply {
            text = s
            setTextSize(TypedValue.COMPLEX_UNIT_SP, size)
            setTextColor(color)
            typeface = when {
                mono -> this@Kit.mono
                bold -> medium
                else -> Typeface.DEFAULT
            }
            includeFontPadding = false
            setLineSpacing(0f, 1.15f)
        }

    /** Small upper-case section label. */
    fun label(s: String) = text(s.uppercase(), 11f, Palette.TEXT3, bold = true).apply { letterSpacing = 0.1f }

    enum class Style { PRIMARY, TEAL, DANGER, SECONDARY, GHOST }

    fun button(label: CharSequence, style: Style = Style.SECONDARY, sizeSp: Float = 15f, onClick: () -> Unit) = TextView(ctx).apply {
        text = label
        gravity = Gravity.CENTER
        setTextSize(TypedValue.COMPLEX_UNIT_SP, sizeSp)
        typeface = medium
        isClickable = true
        isFocusable = true
        minHeight = dp(if (style == Style.GHOST) 40 else 52)
        setPadding(dp(14), dp(8), dp(14), dp(8))
        val (fill, stroke, fg) = when (style) {
            Style.PRIMARY -> Triple(Palette.BLUE, null, 0xFFFFFFFF.toInt())
            Style.TEAL -> Triple(0x221FC9A5, Palette.TEAL, Palette.TEAL)
            Style.DANGER -> Triple(0x22FF5C5C, Palette.RED, Palette.RED)
            Style.SECONDARY -> Triple(Palette.SURFACE2, Palette.STROKE, Palette.TEXT)
            Style.GHOST -> Triple(0x00000000, null, Palette.TEXT2)
        }
        setTextColor(fg)
        background = ripple(rounded(fill, 14f, stroke), 14f)
        setOnClickListener { onClick() }
    }

    /** Enabled look: disabled buttons fade instead of disappearing, so the layout never jumps. */
    fun enable(v: View, on: Boolean) {
        v.isEnabled = on
        v.alpha = if (on) 1f else 0.35f
    }

    /** Status chip "● FLIR" with a coloured dot. */
    fun chip(text: String, dot: Int?) = TextView(ctx).apply {
        maxLines = 1
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f)
        setTextColor(Palette.TEXT2)
        typeface = medium
        setPadding(dp(10), dp(6), dp(12), dp(6))
        background = rounded(Palette.SURFACE2, 999f)
        setChip(this, text, dot)
    }

    fun setChip(v: TextView, text: String, dot: Int?) {
        val sb = SpannableStringBuilder()
        if (dot != null) {
            sb.append("● ")
            sb.setSpan(ForegroundColorSpan(dot), 0, 1, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
        sb.append(text)
        v.text = sb
    }

    fun card(padDp: Int = 14, fill: Int = Palette.SURFACE) = LinearLayout(ctx).apply {
        orientation = LinearLayout.VERTICAL
        background = rounded(fill, 18f)
        setPadding(dp(padDp), dp(padDp), dp(padDp), dp(padDp))
    }

    fun row(gravity: Int = Gravity.CENTER_VERTICAL) = LinearLayout(ctx).apply {
        orientation = LinearLayout.HORIZONTAL
        this.gravity = gravity
    }

    fun col() = LinearLayout(ctx).apply { orientation = LinearLayout.VERTICAL }

    fun lp(w: Int = ViewGroup.LayoutParams.MATCH_PARENT, h: Int = ViewGroup.LayoutParams.WRAP_CONTENT, weight: Float = 0f,
           l: Int = 0, t: Int = 0, r: Int = 0, b: Int = 0) =
        LinearLayout.LayoutParams(w, h, weight).apply { setMargins(dp(l), dp(t), dp(r), dp(b)) }

    fun wrap() = ViewGroup.LayoutParams.WRAP_CONTENT
    fun match() = ViewGroup.LayoutParams.MATCH_PARENT

    fun space(wDp: Int = 0, hDp: Int = 0) = View(ctx).apply { layoutParams = LinearLayout.LayoutParams(dp(wDp), dp(hDp)) }

    /** Clips a view (an image) to rounded corners. */
    fun clipRounded(v: View, radiusDp: Float) {
        v.outlineProvider = object : ViewOutlineProvider() {
            override fun getOutline(view: View, outline: Outline) = outline.setRoundRect(0, 0, view.width, view.height, dp(radiusDp).toFloat())
        }
        v.clipToOutline = true
    }

    /** Segmented control; [onSelect] gets the index. [disabled] options are faded and only call [onDisabledTap]. */
    fun segmented(options: List<String>, selected: Int, disabled: Set<Int> = emptySet(), onDisabledTap: (Int) -> Unit = {},
                  onSelect: (Int) -> Unit): LinearLayout {
        val box = row().apply {
            background = rounded(Palette.SURFACE2, 12f, Palette.STROKE)
            setPadding(dp(3), dp(3), dp(3), dp(3))
        }
        val items = ArrayList<TextView>()
        fun paint(sel: Int) {
            for ((i, t) in items.withIndex()) {
                t.background = if (i == sel) rounded(Palette.BLUE, 10f) else null
                t.setTextColor(if (i == sel) 0xFFFFFFFF.toInt() else Palette.TEXT2)
            }
        }
        for ((i, o) in options.withIndex()) {
            val t = text(o, 13.5f, Palette.TEXT2, bold = true).apply {
                gravity = Gravity.CENTER
                setPadding(dp(8), dp(9), dp(8), dp(9))
                isClickable = true
                if (i in disabled) {
                    alpha = 0.35f
                    setOnClickListener { onDisabledTap(i) }
                } else {
                    setOnClickListener {
                        paint(i)
                        onSelect(i)
                    }
                }
            }
            items += t
            box.addView(t, lp(0, wrap(), 1f))
        }
        paint(selected)
        return box
    }

    /** A segmented control showing a fixed choice: greyed, the fixed option in grey, nothing to tap. */
    fun fixedSegmented(options: List<String>, selected: Int): LinearLayout {
        val box = row().apply {
            background = rounded(Palette.SURFACE2, 12f, Palette.STROKE)
            setPadding(dp(3), dp(3), dp(3), dp(3))
            alpha = 0.65f
        }
        for ((i, o) in options.withIndex()) {
            val t = text(o, 13f, if (i == selected) Palette.TEXT2 else Palette.TEXT3, bold = true).apply {
                gravity = Gravity.CENTER
                setPadding(dp(8), dp(7), dp(8), dp(7))
                if (i == selected) background = rounded(0xFF3B4558.toInt(), 10f)
            }
            box.addView(t, lp(0, wrap(), 1f))
        }
        return box
    }

    /** A text box showing a fixed value: greyed, not editable. */
    fun fixedField(value: String) = field(value).apply {
        isEnabled = false
        isFocusable = false
        setTextColor(Palette.TEXT3)
        setPadding(dp(12), dp(8), dp(12), dp(8))
        alpha = 0.65f
    }

    fun field(value: String, numeric: Boolean = false, hint: String = "") = EditText(ctx).apply {
        setText(value)
        this.hint = hint
        setHintTextColor(Palette.TEXT3)
        setTextColor(Palette.TEXT)
        setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
        background = rounded(Palette.SURFACE2, 12f, Palette.STROKE)
        setPadding(dp(12), dp(11), dp(12), dp(11))
        isSingleLine = true
        if (numeric) inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL or InputType.TYPE_NUMBER_FLAG_SIGNED
    }

    /** Title + subtitle on the left, a switch on the right. */
    fun switchRow(title: String, subtitle: String, checked: Boolean, onChange: (Boolean) -> Unit): Pair<LinearLayout, Switch> {
        val r = row()
        val texts = col()
        texts.addView(text(title, 15f))
        if (subtitle.isNotEmpty()) texts.addView(text(subtitle, 12.5f, Palette.TEXT2), lp(t = 3))
        r.addView(texts, lp(0, wrap(), 1f))
        val sw = Switch(ctx).apply {
            isChecked = checked
            setOnCheckedChangeListener { _, b -> onChange(b) }
        }
        r.addView(sw, lp(wrap(), wrap(), l = 8))
        return r to sw
    }

    /** Keeps a full-screen root clear of the status / navigation bars (Android 15 draws edge to edge). */
    fun padForSystemBars(root: View) {
        root.setOnApplyWindowInsetsListener { v, insets ->
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                val b = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.displayCutout())
                v.setPadding(b.left, b.top, b.right, b.bottom)
            } else {
                @Suppress("DEPRECATION")
                v.setPadding(insets.systemWindowInsetLeft, insets.systemWindowInsetTop, insets.systemWindowInsetRight, insets.systemWindowInsetBottom)
            }
            insets
        }
    }

    fun frame() = FrameLayout(ctx)
}

/** Thin rounded progress line; [fraction] null = indeterminate (a sliding segment). */
class ProgressLine(ctx: Context, private val colour: Int = Palette.BLUE) : View(ctx) {
    private val track = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Palette.SURFACE2 }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = colour }
    private val rect = RectF()
    private var phase = 0f
    private val anim = ValueAnimator.ofFloat(0f, 1f).apply {
        duration = 1200
        repeatCount = ValueAnimator.INFINITE
        addUpdateListener {
            phase = it.animatedValue as Float
            invalidate()
        }
    }

    var fraction: Double? = 0.0
        set(value) {
            field = value
            if (value == null && !anim.isStarted) anim.start()
            if (value != null && anim.isStarted) anim.cancel()
            invalidate()
        }

    var color: Int
        get() = fill.color
        set(value) {
            fill.color = value
            invalidate()
        }

    override fun onDraw(canvas: Canvas) {
        val r = height / 2f
        rect.set(0f, 0f, width.toFloat(), height.toFloat())
        canvas.drawRoundRect(rect, r, r, track)
        val f = fraction
        if (f == null) {
            val segment = width * 0.3f
            val x = -segment + (width + segment) * phase
            rect.set(maxOf(0f, x), 0f, minOf(width.toFloat(), x + segment), height.toFloat())
        } else {
            rect.set(0f, 0f, (width * f.coerceIn(0.0, 1.0)).toFloat(), height.toFloat())
        }
        if (rect.width() > 0) canvas.drawRoundRect(rect, r, r, fill)
    }

    override fun onDetachedFromWindow() {
        anim.cancel()
        super.onDetachedFromWindow()
    }
}

/** Shows its bitmap across the full width, centred vertically (cropped top / bottom when the view is shorter). */
class FitWidthImageView(ctx: Context) : ImageView(ctx) {
    init {
        scaleType = ScaleType.MATRIX
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        fit()
    }

    override fun setImageBitmap(bm: android.graphics.Bitmap?) {
        super.setImageBitmap(bm)
        fit()
    }

    private fun fit() {
        val d = drawable ?: return
        if (width == 0 || d.intrinsicWidth <= 0) return
        val scale = width.toFloat() / d.intrinsicWidth
        imageMatrix = android.graphics.Matrix().apply {
            setScale(scale, scale)
            postTranslate(0f, (height - d.intrinsicHeight * scale) / 2f)
        }
    }
}

/** A ring with a turning arc (the connecting popup); its look does not depend on where the animation is. */
class Spinner(ctx: Context, private val colour: Int = Palette.BLUE) : View(ctx) {
    private val stroke = 4f * ctx.resources.displayMetrics.density
    private val track = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = stroke
        color = (colour and 0x00FFFFFF) or 0x33000000
    }
    private val arc = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = stroke
        strokeCap = Paint.Cap.ROUND
        color = colour
    }
    private val rect = RectF()
    private var angle = 0f
    private val anim = ValueAnimator.ofFloat(0f, 360f).apply {
        duration = 900
        repeatCount = ValueAnimator.INFINITE
        interpolator = android.view.animation.LinearInterpolator()
        addUpdateListener {
            angle = it.animatedValue as Float
            invalidate()
        }
    }

    override fun onAttachedToWindow() {
        super.onAttachedToWindow()
        anim.start()
    }

    override fun onDetachedFromWindow() {
        anim.cancel()
        super.onDetachedFromWindow()
    }

    override fun onDraw(canvas: Canvas) {
        val h = stroke / 2
        rect.set(h, h, width - h, height - h)
        canvas.drawOval(rect, track)
        canvas.drawArc(rect, angle - 90f, 100f, false, arc)
    }
}

/**
 * Crosshair over an [ImageView] at a point given as fractions of the shown image (the sun-mode aim mark at
 * the optical axis), following the image's FIT_CENTER placement.
 */
class Crosshair(ctx: Context, private val image: ImageView) : View(ctx) {
    var fx = 0.5f
    var fy = 0.5f
    private val dp = ctx.resources.displayMetrics.density
    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 2f * dp
        color = 0xFFFF5252.toInt()
    }
    private val shadow = Paint(line).apply {
        strokeWidth = 4f * dp
        color = 0x99000000.toInt()
    }

    override fun onDraw(canvas: Canvas) {
        val d = image.drawable ?: return
        val m = android.graphics.Matrix(image.imageMatrix)
        val p = floatArrayOf(fx * d.intrinsicWidth, fy * d.intrinsicHeight)
        m.mapPoints(p)
        val x = p[0] + image.left + image.paddingLeft
        val y = p[1] + image.top + image.paddingTop
        val r = 14f * dp
        val gap = 5f * dp
        for (paint in listOf(shadow, line)) {
            canvas.drawCircle(x, y, r, paint)
            canvas.drawLine(x - r - 10 * dp, y, x - gap, y, paint)
            canvas.drawLine(x + gap, y, x + r + 10 * dp, y, paint)
            canvas.drawLine(x, y - r - 10 * dp, x, y - gap, paint)
            canvas.drawLine(x, y + gap, x, y + r + 10 * dp, paint)
        }
    }
}
