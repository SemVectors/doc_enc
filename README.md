Библиотека ExactusSemVectors для формирования кросс-языковых векторных представлений текстов и их фрагментов на основе глубокого обучения для решения задач информационного поиска и классификации текстовой информации.

[Документация](https://semvectors-doc-enc.readthedocs.io/ru/latest/index.html)


--------------------------------->cut here<-------------------------------------

Открытая библиотека ExactusSemVectors для формирования кросс-языковых векторных представлений текстов и их фрагментов на основе глубокого обучения может применяться в различных областях для решения задач семантического анализа естественно-языковых текстов,
в которых требуется преобразование длинного текста в векторное представление (эмбеддинги).
К числу таких задач относятся:
- сравнение документов по смыслу (semantic matching);
- задачи информационного поиска, в которых в качестве запроса выступает документ-образец, для которого необходимо найти похожие документы;
- текстовая классификация и кластеризация.

Сферой применения библиотеки могут быть промышленные наукоёмкие решения:
- поисково-аналитические машины,
- DLP-системы,
- системы текстовой аналитики,
- интеллектуального подбора кадров,
- динамической контентной фильтрации.


Минимальные технические требования для запуска и использования открытой библиотеки
- Процессор: 
  + архитектура: x86_64 (не менее);
  + количество ядер: 4 (не менее) (для обучения 16 не менее);
  + тактовая частота процессора 2.4 Ггц (не менее).
- Оперативная память 32 Гбайт, не менее.
- Дисковая подсистема объемом не менее 500 Гбайт:
  + скорость записи не менее 100 Мб/с,
  + скорость чтения не менее 100 Мб/с.
- Графический вычислитель (GPU) с объёмом оперативной памяти (видеопамяти) 6 Гбайт или более (для обучения 32 Гб или более).
- Операционная система:  ОС семейства Linux.
