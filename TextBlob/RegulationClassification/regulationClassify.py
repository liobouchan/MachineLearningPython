from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob

train = [
    ('se incluirán', 'Obligación'),
    ('se considerará', 'Obligación'),
    ('deberán', 'Obligación'),
    ('deberá', 'Obligación'),
    ('deberá establecer y mantener', 'Obligación'),
    ('se considerarán', 'Obligación'),
    ('se efectuará', 'Obligación'),
    ('se realizará considerando', 'Obligación'),
    ('se determinarán', 'Obligación'),
    ('Efectuarán', 'Obligación'),
    ('Deberán proporcionar', 'Obligación'),
    ('Deberán proporcionarla', 'Obligación'),
    ('Deberán proporcionarlo', 'Obligación'),
    ('Tendrá que', 'Obligación'),
    ('Se efectuará', 'Obligación'),
    ('Deberán mantener', 'Obligación'),
    ('No podrá', 'Obligación'),
    ('Estará', 'Obligación'),
    ('Deben mantener', 'Obligación'),
    ('Se integrará', 'Obligación'),
    ('Se considerarán', 'Obligación'),
    ('Deberán considerar', 'Obligación'),
    ('computarán', 'Obligación'),
    #Ajustes de Obligaciones No es bueno hacer ajustes
    ('Para efectos del reconocimiento de Operaciones, se incluirán las Operaciones a partir de la fecha en que se concierten' , 'Obligación'),


    ('podrá', 'Recomendación'),
    ('se podrán', 'Recomendación'),
    ('podrá aplicarse', 'Recomendación'),
    ('se podrán aplicar', 'Recomendación'),

    # #('para el caso deberán ser', 'Requerimiento'),
    # ('para el caso en que las Instituciones cuenten con Modelos de Valuación Internos,las posiciones valuadas mediante dichos modelos deberán ser ajustadas', 'Requerimiento'),
    # #('tratandose de deberán establecer y mantener', 'Requerimiento'),
    # ('tratandose de deberán considerar', 'Requerimiento'),
    # #('en el caso de se considerará', 'Requerimiento'),
    # #('en el caso de se considerarán', 'Requerimiento'),
    # ('se sumarán', 'Requerimiento'),
    # ('se realizará', 'Requerimiento'),
    # ('se convertirá', 'Requerimiento'),
]

regulationClassifier = NaiveBayesClassifier(train)

textBlobParagraph = TextBlob("Artículo 2 Bis Para tales efectos, tratándose del riesgo de crédito podrá aplicarse alguno de los dos enfoques, un Método Estándar, al cual se refiere la Sección Segunda del Capítulo III del presente título, y otro basado en calificaciones internas, este último de tipo básico o avanzado, cuyo uso estará sujeto a lo dispuesto en la Sección Tercera del citado Capítulo III. "
                             "Artículo 2 Bis En lo que se refiere al riesgo de mercado, las Instituciones utilizarán el método estándar. Para el Riesgo Operacional se podrán aplicar distintos métodos de complejidad creciente conforme a lo que se establece en el presente título. "
                             "Artículo 2 Bis 1.- Para efectos del reconocimiento de Operaciones, se incluirán las Operaciones a partir de la fecha en que se concierten, independientemente de la fecha de liquidación, entrega o vigencia, según sea el caso. "
                             "Artículo 2 Bis 1.- Se considerará que se ha transferido la propiedad de un activo, y que por lo tanto éste no tendrá requerimientos de capitalización de acuerdo con lo establecido en el presente título, siempre que se cumpla con los Criterios Contables. "
                             "Artículo 2 Bis 2.- En la determinación del importe de las operaciones para los efectos del presente título, las Operaciones deberán ser valuadas conforme a los Criterios Contables. "
                             "Artículo 2 Bis 2.- Asimismo, para el caso en que las Instituciones cuenten con Modelos de Valuación Internos, las posiciones valuadas mediante dichos modelos deberán ser ajustadas para efectos del presente título considerando como mínimo los costos de cancelación y cierre de posiciones, los riesgos operacionales, los costos de financiamiento de las operaciones, los gastos administrativos futuros, diferenciales crediticios no reconocidos, el riesgo del modelo, incluyendo aquel asociado con el uso de una metodología incorrecta de valuación y con la calibración de parámetros no observables, así como, en su caso, la iliquidez de las posiciones. "
                             "Artículo 2 Bis 2.- Tratándose de aquellas posiciones ilíquidas, las Instituciones deberán establecer y mantener procedimientos para identificarlas así como revisar continuamente que el ajuste correspondiente sigue siendo vigente, para lo cual deberán considerar factores tales como el tiempo que se necesitaría para cubrir la posición, la volatilidad de los diferenciales entre los precios de compra y venta, la disponibilidad de cotizaciones de mercado, el promedio y volatilidad de los montos que se comercializan tanto en condiciones normales como en condiciones de estrés, concentraciones de mercado y el tiempo transcurrido desde la concertación de las operaciones. "
                             "Artículo 2 Bis 2.- En el caso del Método Estándar a que se refiere el presente título, la cartera de créditos se considerará neta de las correspondientes reservas crediticias constituidas que no computen como capital complementario en términos de lo dispuesto en la fracción III del Artículo 2 Bis 7 de estas disposiciones; los valores y otros activos, en su caso, se considerarán netos de las respectivas estimaciones, depreciaciones y castigos. "
                             "Artículo 2 Bis 3.- Las Operaciones de las entidades financieras del exterior a que se refiere el segundo párrafo del Artículo 89 de la Ley, para todos los efectos de lo dispuesto en el Artículo 73 de la mencionada Ley se considerarán como realizadas por la propia Institución. "
                             "Artículo 2 Bis 3.- Asimismo, las Operaciones de estas entidades del exterior, para efectos de lo previsto en el presente título, se considerarán conforme a lo siguiente: "
                             "Artículo 2 Bis 3.- I. Se efectuará un cómputo de requerimientos de capital para cada entidad financiera filial del exterior, aplicando lo dispuesto en el presente título al total de las Operaciones de éstas, y"
                             "Artículo 2 Bis 3.- II. En caso de que el requerimiento de capital obtenido conforme al inciso anterior sea superior al importe del capital neto de la entidad financiera del exterior de que se trate, la diferencia entre ambas cantidades se sumará para todos los efectos a los requerimientos de capital de la Institución."
    , classifier = regulationClassifier)

for sentence in textBlobParagraph.sentences:
    #print("Enunciado : " , sentence)
    print("  Tipo : " , sentence.classify())
    print("  Palabras Clave : " , sentence.noun_phrases)
    print("      ")

regulationClassifier.show_informative_features(5)