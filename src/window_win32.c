/* window_win32.c - Win32 window, input and GDI presentation. */
#include "window.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>

struct Window {
    HWND  hwnd;
    HDC   hdc;
    int   w, h;
    int   open;
    unsigned char down[256], pressed[256];
    int   capture, cursor_hidden;
    POINT center;
    int   mdx, mdy, wheel;
};

static void set_capture(Window* w, int on) {
    if (on == w->capture) return;
    w->capture = on;
    if (on) {
        RECT rc;
        GetClientRect(w->hwnd, &rc);
        w->center.x = (rc.right - rc.left) / 2;
        w->center.y = (rc.bottom - rc.top) / 2;
        ClientToScreen(w->hwnd, &w->center);
        SetCapture(w->hwnd);
        SetCursorPos(w->center.x, w->center.y);
        if (!w->cursor_hidden) { ShowCursor(FALSE); w->cursor_hidden = 1; }
    } else {
        ReleaseCapture();
        if (w->cursor_hidden) { ShowCursor(TRUE); w->cursor_hidden = 0; }
    }
}

static LRESULT CALLBACK wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    Window* w = (Window*)GetWindowLongPtrA(hwnd, GWLP_USERDATA);
    if (w) {
        switch (msg) {
        case WM_CLOSE: w->open = 0; return 0;
        case WM_SIZE: w->w = LOWORD(lp); w->h = HIWORD(lp); return 0;
        case WM_ERASEBKGND: return 1;
        case WM_KEYDOWN:
            if (wp < 256) { if (!w->down[wp]) w->pressed[wp] = 1; w->down[wp] = 1; }
            return 0;
        case WM_KEYUP:
            if (wp < 256) w->down[wp] = 0;
            return 0;
        case WM_KILLFOCUS:
            memset(w->down, 0, sizeof w->down);
            set_capture(w, 0);
            return 0;
        case WM_RBUTTONDOWN: set_capture(w, 1); return 0;
        case WM_RBUTTONUP:   set_capture(w, 0); return 0;
        case WM_MOUSEWHEEL:  w->wheel += GET_WHEEL_DELTA_WPARAM(wp); return 0;
        default: break;
        }
    }
    return DefWindowProcA(hwnd, msg, wp, lp);
}

Window* window_create(const char* title, int width, int height) {
    SetProcessDPIAware();
    HINSTANCE inst = GetModuleHandleA(NULL);
    WNDCLASSA wc;
    memset(&wc, 0, sizeof wc);
    wc.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = wndproc;
    wc.hInstance = inst;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = "GMEWindow";
    RegisterClassA(&wc);

    RECT r = { 0, 0, width, height };
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowA("GMEWindow", title, WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                              r.right - r.left, r.bottom - r.top, NULL, NULL, inst, NULL);
    if (!hwnd) return NULL;
    Window* w = (Window*)calloc(1, sizeof *w);
    w->hwnd = hwnd;
    w->hdc = GetDC(hwnd);
    w->open = 1;
    SetWindowLongPtrA(hwnd, GWLP_USERDATA, (LONG_PTR)w);
    SetStretchBltMode(w->hdc, COLORONCOLOR);
    RECT rc;
    GetClientRect(hwnd, &rc);
    w->w = rc.right - rc.left;
    w->h = rc.bottom - rc.top;
    ShowWindow(hwnd, SW_SHOW);
    SetForegroundWindow(hwnd);
    return w;
}

void window_destroy(Window* w) {
    if (!w) return;
    set_capture(w, 0);
    ReleaseDC(w->hwnd, w->hdc);
    DestroyWindow(w->hwnd);
    free(w);
}

int window_pump(Window* w) {
    memset(w->pressed, 0, sizeof w->pressed);
    MSG m;
    while (PeekMessageA(&m, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&m);
        DispatchMessageA(&m);
    }
    if (w->capture) {
        POINT p;
        GetCursorPos(&p);
        w->mdx += p.x - w->center.x;
        w->mdy += p.y - w->center.y;
        SetCursorPos(w->center.x, w->center.y);
    }
    return w->open;
}

void window_client_size(const Window* w, int* width, int* height) { *width = w->w; *height = w->h; }

void window_present(Window* w, const uint32_t* px, int width, int height) {
    if (w->w <= 0 || w->h <= 0) return; /* minimized */
    BITMAPINFO bmi;
    memset(&bmi, 0, sizeof bmi);
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height; /* top-down */
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    StretchDIBits(w->hdc, 0, 0, w->w, w->h, 0, 0, width, height, px, &bmi, DIB_RGB_COLORS, SRCCOPY);
}

void window_set_title(Window* w, const char* t) { SetWindowTextA(w->hwnd, t); }
int  window_key_down(const Window* w, int k) { return k >= 0 && k < 256 && w->down[k]; }
int  window_key_pressed(const Window* w, int k) { return k >= 0 && k < 256 && w->pressed[k]; }

void window_mouse_delta(Window* w, int* dx, int* dy) {
    *dx = w->mdx; *dy = w->mdy;
    w->mdx = w->mdy = 0;
}

int window_wheel(Window* w) {
    const int n = w->wheel / WHEEL_DELTA;
    w->wheel -= n * WHEEL_DELTA;
    return n;
}
#endif

