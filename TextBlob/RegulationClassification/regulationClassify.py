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
    ('se efectará', 'Obligación'),
    ('Deberán mantener', 'Obligación'),
    ('No podrá', 'Obligación'),
    ('Estará', 'Obligación'),
    ('Deben mantener', 'Obligación'),
    ('Se integrará', 'Obligación'),
    ('Se considerarán', 'Obligación'),
    ('Deberán considerar', 'Obligación'),
    ('computarán', 'Obligación'),
    ('podrá', 'Recomendación'),
    ('se podrán', 'Recomendación'),
    ('podrá aplicarse', 'Recomendación'),
    ('se podrán aplicar', 'Recomendación'),
    ('para el caso x deberán ser', 'Requerimiento'),
    ('tratándose de x deberán establecer y mantener', 'Requerimiento'),
    ('en el caso de x se considerará', 'Requerimiento'),
    ('en el caso de x se considerarán', 'Requerimiento'),
    ('en caso de que x se sumarán', 'Requerimiento'),
    ('dólares de los Estados Unidos de América se realizará', 'Requerimiento'),
    ('UDIs se realizará', 'Requerimiento'),
    ('se convertirá' , 'Requerimiento'),
    ('tratándose de x deberán mantener', 'Requerimiento')
]

cl = NaiveBayesClassifier(train)

print(cl.classify(""))