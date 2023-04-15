# INF8215-TP3 Phishing (Équipe MMA)
Travail réalisé par:
- Manu Serpette 2229693
- Marc-Antoine Bettez 1828113
- Allen Yu 1958185

## Requirements
Pour installer les packages nécessaires lancer la commande suivante:
```
    pip install -r .\requirements.txt
```

## Utilisation
Pour lancer le GridSearch, l'entrainement et la création du fichier de submission:
```
    python train.py {PATH_DATASET} {PATH_TEST_DATASET}
```
Exemple:
```
    python .\train.py .\train.csv .\test.csv
```
Suite à l'exéction, un fichier submission.csv sera créé.
