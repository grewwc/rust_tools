package com.grewwc.aserve;

import android.animation.ValueAnimator;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.provider.Settings;
import android.util.DisplayMetrics;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;
import android.view.WindowManager;

/**
 * A small floating capture button drawn above other apps.
 *
 * <p>The capture trigger used to live in the notification shade, which covers
 * the content the user wants to share: the frame that arrives when the shade
 * button is tapped contains the shade, and so does any frame taken just
 * before the tap. The bubble floats above the content instead, so the frame
 * that is delivered is the one the user was looking at when they tapped.</p>
 *
 * <p>Drawing an overlay needs the "display over other apps" permission, which
 * the user grants once in the system settings. Without it the service keeps
 * using the notification action, so the feature still works.</p>
 */
final class ScreenBubble {
    private static final String PREFS = "screen_bubble";
    private static final String KEY_X = "x";
    private static final String KEY_Y = "y";
    /** Edge length of the bubble in density independent pixels. */
    private static final int SIZE_DP = 52;
    private static final int ACCENT = 0xFF2E8BFF;
    private static final int FACE = 0xE61B2129;

    private final Context context;
    private final WindowManager windows;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final BubbleView view;
    private final BubbleTouch touch = new BubbleTouch();
    private final int slop;
    private final WindowManager.LayoutParams params;

    /** What the caller wants; {@link #applied} is what the screen shows. */
    private boolean desired;
    private boolean applied;
    private Runnable onTap;
    private float downRawX;
    private float downRawY;
    private int startX;
    private int startY;
    /** True while this class removes the view, so the detach callback ignores it. */
    private boolean detaching;

    ScreenBubble(Context context) {
        this.context = context.getApplicationContext();
        this.windows = (WindowManager) this.context.getSystemService(Context.WINDOW_SERVICE);
        this.slop = ViewConfiguration.get(this.context).getScaledTouchSlop();
        this.view = new BubbleView(this.context);
        this.view.setOnTouchListener(this::onTouch);
        this.view.setOnDetach(this::onDetached);
        this.params = buildParams();
    }

    /** True when the user granted the "display over other apps" permission. */
    static boolean permitted(Context context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M) return true;
        return Settings.canDrawOverlays(context);
    }

    /**
     * Opens the system page where the overlay permission is granted. The grant
     * survives restarts, so this is a one-time step.
     */
    static void openSettings(Context context) {
        Intent intent = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                Uri.parse("package:" + context.getPackageName()));
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        try {
            context.startActivity(intent);
            return;
        } catch (RuntimeException ignored) {
            // Some ROMs do not offer the per-app page; the list page still works.
        }
        Intent list = new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION);
        list.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        try {
            context.startActivity(list);
        } catch (RuntimeException ignored) {
            // No settings page on this device: the bubble simply stays unavailable.
        }
    }

    private WindowManager.LayoutParams buildParams() {
        int size = (int) (SIZE_DP * context.getResources().getDisplayMetrics().density);
        WindowManager.LayoutParams layout = new WindowManager.LayoutParams(size, size,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                        | WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL
                        | WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
                PixelFormat.TRANSLUCENT);
        layout.gravity = Gravity.TOP | Gravity.START;
        SharedPreferences prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE);
        layout.x = prefs.getInt(KEY_X, Integer.MIN_VALUE);
        layout.y = prefs.getInt(KEY_Y, Integer.MIN_VALUE);
        if (layout.x == Integer.MIN_VALUE || layout.y == Integer.MIN_VALUE) {
            DisplayMetrics metrics = context.getResources().getDisplayMetrics();
            layout.x = Math.max(0, metrics.widthPixels - size - dp(10));
            layout.y = Math.max(0, (int) (metrics.heightPixels * 0.35f));
        }
        return layout;
    }

    void setOnTap(Runnable onTap) {
        this.onTap = onTap;
    }

    /** Adds the bubble to the screen. Returns false when the overlay is not allowed. */
    boolean show() {
        desired = true;
        apply();
        return applied;
    }

    /** Removes the bubble from the screen. */
    void hide() {
        desired = false;
        apply();
    }

    boolean visible() {
        return applied;
    }

    /** Highlights the bubble while a capture request waits for the user. */
    void setArmed(boolean armed) {
        if (Looper.myLooper() != Looper.getMainLooper()) {
            main.post(() -> setArmed(armed));
            return;
        }
        view.setArmed(armed);
    }

    /**
     * Hides the bubble for the moment the frame is taken, so the delivered
     * picture shows the content and not the bubble that floats above it.
     */
    void setDrawn(boolean drawn) {
        if (Looper.myLooper() != Looper.getMainLooper()) {
            main.post(() -> setDrawn(drawn));
            return;
        }
        if (applied) view.setVisibility(drawn ? View.VISIBLE : View.INVISIBLE);
    }

    /**
     * Brings the view in line with the wanted state. Adding and removing a
     * window is main-thread work, and callers may reach this from a WebView
     * thread, so the work is moved there when needed.
     */
    private void apply() {
        if (Looper.myLooper() != Looper.getMainLooper()) {
            main.post(this::apply);
            return;
        }
        if (desired == applied) return;
        if (!desired) {
            detaching = true;
            try {
                windows.removeView(view);
            } catch (RuntimeException ignored) {
                // The view was already detached; nothing left to do.
            } finally {
                detaching = false;
            }
            applied = false;
            return;
        }
        try {
            windows.addView(view, params);
            applied = true;
        } catch (RuntimeException error) {
            // The overlay permission was revoked while the service was running.
            applied = false;
            desired = false;
        }
    }

    /**
     * The system can take the overlay down on its own, for example when the
     * app moves to the background. The window is then gone even though the
     * request is still armed, so the wanted state has to be applied again.
     */
    private void onDetached() {
        if (detaching) return;
        applied = false;
    }

    private int dp(int value) {
        return (int) (value * context.getResources().getDisplayMetrics().density);
    }

    private boolean onTouch(View touched, MotionEvent event) {
        switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                touch.down(event.getRawX(), event.getRawY(), event.getEventTime());
                downRawX = event.getRawX();
                downRawY = event.getRawY();
                startX = params.x;
                startY = params.y;
                return true;
            case MotionEvent.ACTION_MOVE:
                if (touch.moved(event.getRawX(), event.getRawY(), slop)) {
                    params.x = startX + (int) (event.getRawX() - downRawX);
                    params.y = startY + (int) (event.getRawY() - downRawY);
                    try {
                        windows.updateViewLayout(view, params);
                    } catch (RuntimeException ignored) {
                        // The view was detached mid-drag; the position is kept.
                    }
                }
                return true;
            case MotionEvent.ACTION_UP:
                if (touch.up(event.getRawX(), event.getRawY(), event.getEventTime(), slop)) {
                    if (onTap != null) onTap.run();
                } else {
                    savePosition();
                }
                return true;
            case MotionEvent.ACTION_CANCEL:
                touch.cancel();
                return true;
            default:
                return false;
        }
    }

    /** Remembers where the user put the bubble, so it stays there next time. */
    private void savePosition() {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putInt(KEY_X, params.x)
                .putInt(KEY_Y, params.y)
                .apply();
    }

    /** The bubble itself: a camera glyph on a floating disc. */
    private static final class BubbleView extends View {
        private final Paint fill = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint ring = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint glyph = new Paint(Paint.ANTI_ALIAS_FLAG);
        private boolean armed;
        private float pulse;
        private Runnable onDetach;

        BubbleView(Context context) {
            super(context);
            setFocusable(false);
            setClickable(true);
        }

        void setOnDetach(Runnable listener) { onDetach = listener; }

        @Override protected void onDetachedFromWindow() {
            super.onDetachedFromWindow();
            Runnable listener = onDetach;
            if (listener != null) listener.run();
        }

        void setArmed(boolean armed) {
            if (this.armed == armed) return;
            this.armed = armed;
            if (armed) startPulse();
            else {
                pulse = 0f;
                invalidate();
            }
        }

        private void startPulse() {
            ValueAnimator animator = ValueAnimator.ofFloat(0f, 1f);
            animator.setDuration(1100L);
            animator.setRepeatCount(ValueAnimator.INFINITE);
            animator.setRepeatMode(ValueAnimator.REVERSE);
            animator.addUpdateListener(value -> {
                pulse = (float) value.getAnimatedValue();
                invalidate();
            });
            animator.start();
        }

        @Override
        protected void onDraw(Canvas canvas) {
            float size = getWidth();
            float r = size / 2f;
            fill.setColor(armed ? ACCENT : FACE);
            canvas.drawCircle(r, r, r - 1f, fill);
            ring.setStyle(Paint.Style.STROKE);
            ring.setStrokeWidth(size * 0.045f);
            ring.setColor(armed ? 0x88FFFFFF : 0x40FFFFFF);
            canvas.drawCircle(r, r, r - size * 0.07f, ring);
            if (armed && pulse > 0f) {
                ring.setColor(0x26FFFFFF);
                canvas.drawCircle(r, r, (r - size * 0.07f) * (1f + pulse * 0.3f), ring);
            }
            drawCamera(canvas, r, size);
        }

        private void drawCamera(Canvas canvas, float r, float size) {
            glyph.setStyle(Paint.Style.FILL);
            glyph.setColor(0xFFFFFFFF);
            float width = size * 0.44f;
            float height = size * 0.30f;
            float top = r - height / 2f + size * 0.05f;
            RectF body = new RectF(r - width / 2f, top, r + width / 2f, top + height);
            canvas.drawRoundRect(body, size * 0.06f, size * 0.06f, glyph);
            // The viewfinder bump on top of the camera body.
            canvas.drawRoundRect(new RectF(r - width * 0.24f, top - size * 0.07f,
                    r + width * 0.02f, top + size * 0.01f), size * 0.02f, size * 0.02f, glyph);
            // The lens.
            glyph.setStyle(Paint.Style.STROKE);
            glyph.setStrokeWidth(size * 0.05f);
            canvas.drawCircle(r, body.centerY(), size * 0.085f, glyph);
        }
    }
}
